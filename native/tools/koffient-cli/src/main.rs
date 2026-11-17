//! Development CLI for experimental .koff artifacts.
use koffient_format::{
    encode_layout, pack_onnx, pack_onnx_external, parse_layout, KoffFile, TensorPlacement,
};
use koffient_layout::{plan, Node, Weight};
use std::{collections::BTreeSet, env, error::Error, fs, path::Path, process};

type CliResult<T> = Result<T, Box<dyn Error>>;

fn usage() -> &'static str {
    "Usage:\n\
      koff pack <model.onnx> <model.koff> [name] [alignment=64]\n\
      koff inspect <model.koff>\n\
      koff validate <model.koff>\n\
      koff extract <model.koff> <restored.onnx>\n\
      koff extract-bundle <model.koff> <output-directory>\n\
      koff topo-plan <graph.txt> [alignment=64]\n\
      koff topo-export <graph.txt> <layout.tsv> [alignment=64] [topology|original]\n\
      koff pack-external <model.onnx> <weights.bin> <layout.tsv> <output.koff> [name] [alignment=64]\n\n\
     Topology text format:\n\
      weight <id> <size_bytes>\n\
      node <id> <comma-separated-parent-ids or -> <comma-separated-weight-ids or ->\n\n\
     For pack-external, model.onnx must already reference weights.bin with matching offsets.\n"
}

fn parse_ids(text: &str) -> Result<Vec<u32>, String> {
    if text == "-" {
        return Ok(vec![]);
    }
    text.split(',')
        .map(|s| s.parse::<u32>().map_err(|_| format!("invalid ID '{s}'")))
        .collect()
}

fn read_graph(path: &Path) -> CliResult<(Vec<Node>, Vec<Weight>)> {
    let graph = fs::read_to_string(path)?;
    let mut nodes = vec![];
    let mut weights = vec![];
    for (line_no, line) in graph.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<_> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["weight", id, size] => weights.push(Weight {
                id: id.parse()?,
                size: size.parse()?,
            }),
            ["node", id, parents, ws] => nodes.push(Node {
                id: id.parse()?,
                parents: parse_ids(parents)?,
                weights: parse_ids(ws)?,
            }),
            _ => return Err(format!("line {}: expected weight/node row", line_no + 1).into()),
        }
    }
    Ok((nodes, weights))
}

fn build_placements(
    nodes: &[Node],
    weights: &[Weight],
    alignment: u64,
    policy: &str,
) -> CliResult<Vec<TensorPlacement>> {
    let topo = plan(nodes, weights, alignment)?;
    let rows = if policy == "original" {
        // Tensor IDs are assigned in input initializer order by the ONNX prepare tool.
        let mut items = weights.to_vec();
        items.sort_by_key(|w| w.id);
        let mut cursor = 0u64;
        let mut rows = vec![];
        for w in &items {
            cursor = cursor
                .checked_add(alignment - 1)
                .ok_or("alignment overflow")?
                & !(alignment - 1);
            let first = topo
                .placements
                .iter()
                .find(|x| x.weight_id == w.id)
                .ok_or("missing weight")?
                .first_use_step;
            rows.push(TensorPlacement {
                tensor_id: w.id,
                first_use_step: u32::try_from(first).unwrap_or(u32::MAX),
                offset: cursor,
                length: w.size,
            });
            cursor = cursor.checked_add(w.size).ok_or("size overflow")?;
        }
        rows
    } else if policy == "topology" {
        topo.placements
            .iter()
            .map(|p| TensorPlacement {
                tensor_id: p.weight_id,
                first_use_step: u32::try_from(p.first_use_step).unwrap_or(u32::MAX),
                offset: p.offset,
                length: p.size,
            })
            .collect()
    } else {
        return Err(format!("unknown policy '{policy}': expected topology or original").into());
    };
    Ok(rows)
}

fn read_placements(path: &Path) -> CliResult<Vec<TensorPlacement>> {
    let input = fs::read_to_string(path)?;
    let mut placements = vec![];
    let mut ids = BTreeSet::new();
    for (line_no, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split('\t').collect();
        if parts.len() != 4 {
            return Err(format!(
                "layout.tsv:{}: expected 4 tab-separated columns",
                line_no + 1
            )
            .into());
        }
        let item = TensorPlacement {
            tensor_id: parts[0].parse()?,
            first_use_step: parts[1].parse()?,
            offset: parts[2].parse()?,
            length: parts[3].parse()?,
        };
        if !ids.insert(item.tensor_id) {
            return Err("duplicate tensor id in layout.tsv".into());
        }
        placements.push(item);
    }
    if placements.is_empty() {
        return Err("no tensor placements".into());
    }
    Ok(placements)
}

fn run() -> CliResult<()> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("pack") if (4..=6).contains(&args.len()) => {
            let raw = fs::read(&args[2])?;
            let name = args.get(4).map(String::as_str).unwrap_or("model");
            let align = args
                .get(5)
                .map(String::as_str)
                .unwrap_or("64")
                .parse::<u32>()?;
            let out = pack_onnx(name, &raw, align)?;
            KoffFile::parse(out.clone())?;
            fs::write(&args[3], out)?;
            println!("Created {} (payload {} bytes)", args[3], raw.len());
        }
        Some("pack-external") if (6..=8).contains(&args.len()) => {
            let onnx = fs::read(&args[2])?;
            let weights = fs::read(&args[3])?;
            let rows = read_placements(Path::new(&args[4]))?;
            // Name is argument 6, optional alignment argument 7.
            let name = args.get(6).map(String::as_str).unwrap_or("model");
            let alignment = args
                .get(7)
                .map(String::as_str)
                .unwrap_or("64")
                .parse::<u32>()?;
            let layout = encode_layout(&rows, alignment)?;
            parse_layout(&layout, weights.len() as u64)?;
            let out = pack_onnx_external(name, &onnx, &weights, &layout, alignment)?;
            KoffFile::parse(out.clone())?;
            fs::write(&args[5], out)?;
            println!(
                "Created {} with {} tensor placements and {} weight bytes",
                args[5],
                rows.len(),
                weights.len()
            );
        }
        Some("inspect") if args.len() == 3 => {
            let f = KoffFile::read(&args[2])?;
            println!(
                "KOFF {}.{} | name={} | size={} | sections={}",
                f.header.major,
                f.header.minor,
                f.manifest.model_name,
                f.header.file_size,
                f.sections.len()
            );
            for s in &f.sections {
                println!(
                    "  section: type={} id={} offset={} length={} align={} crc32={:08x}",
                    s.kind, s.id, s.offset, s.length, s.alignment, s.crc32
                );
            }
            if let (Some(layout), Some(weights)) = (f.layout_bytes(), f.external_weights()) {
                let placements = parse_layout(layout, weights.len() as u64)?;
                println!("  external tensors: {}", placements.len());
                for item in placements {
                    println!(
                        "    tensor={} first_step={} offset={} size={}",
                        item.tensor_id, item.first_use_step, item.offset, item.length
                    );
                }
            }
        }
        Some("validate") if args.len() == 3 => {
            let f = KoffFile::read(&args[2])?;
            println!(
                "OK: {} ({}-byte ONNX graph/payload)",
                f.manifest.model_name,
                f.payload().len()
            );
            println!(
                "NOTE: KOFF structure/CRC/layout validated; ONNX graph semantics not validated."
            );
        }
        Some("extract") if args.len() == 4 => {
            let f = KoffFile::read(&args[2])?;
            if f.external_weights().is_some() {
                return Err(
                    "external weights present: use extract-bundle to preserve weights.bin".into(),
                );
            }
            fs::write(&args[3], f.payload())?;
            println!("Extracted {} bytes", f.payload().len());
        }
        Some("extract-bundle") if args.len() == 4 => {
            let f = KoffFile::read(&args[2])?;
            let dir = Path::new(&args[3]);
            fs::create_dir_all(dir)?;
            fs::write(dir.join("model.onnx"), f.payload())?;
            if let Some(w) = f.external_weights() {
                fs::write(dir.join("weights.bin"), w)?;
            }
            println!(
                "Extracted model.onnx{} into {}",
                if f.external_weights().is_some() {
                    " + weights.bin"
                } else {
                    ""
                },
                dir.display()
            );
        }
        Some("topo-plan") if args.len() == 3 || args.len() == 4 => {
            let (nodes, weights) = read_graph(Path::new(&args[2]))?;
            let align = args
                .get(3)
                .map(String::as_str)
                .unwrap_or("64")
                .parse::<u64>()?;
            let p = plan(&nodes, &weights, align)?;
            println!(
                "Execution order: {:?} | tensor arena: {} bytes",
                p.execution_order, p.total_bytes
            );
            for x in p.placements {
                println!(
                    "  weight={} first_use_step={} offset={} length={}",
                    x.weight_id, x.first_use_step, x.offset, x.size
                );
            }
        }
        Some("topo-export") if (4..=6).contains(&args.len()) => {
            let (nodes, weights) = read_graph(Path::new(&args[2]))?;
            let alignment = args
                .get(4)
                .map(String::as_str)
                .unwrap_or("64")
                .parse::<u64>()?;
            let policy = args.get(5).map(String::as_str).unwrap_or("topology");
            let entries = build_placements(&nodes, &weights, alignment, policy)?;
            let mut out = String::from("# tensor_id\tfirst_use_step\toffset\tlength\n");
            for item in &entries {
                out.push_str(&format!(
                    "{}\t{}\t{}\t{}\n",
                    item.tensor_id, item.first_use_step, item.offset, item.length
                ));
            }
            fs::write(&args[3], out)?;
            println!("Wrote {} placements using {} policy", entries.len(), policy);
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(2);
    }
}
