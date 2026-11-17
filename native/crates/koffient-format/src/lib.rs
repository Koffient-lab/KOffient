//! Experimental .koff v0.1 section-based binary container.
//! This crate does NOT execute models or parse ONNX graphs.
use std::fmt;
use std::fs;
use std::path::Path;

pub const HEADER_SIZE: usize = 64;
pub const DIRECTORY_ENTRY_SIZE: usize = 48;
pub const MAX_SECTIONS: usize = 4096;
pub const CRITICAL: u32 = 1;
pub const SECTION_MANIFEST: u32 = 1;
pub const SECTION_MODEL_PAYLOAD: u32 = 2;
/// Raw external tensor data, referenced by offsets in the ONNX graph.
pub const SECTION_EXTERNAL_WEIGHTS: u32 = 3;
/// Binary KTL1 topology layout descriptor.
pub const SECTION_LAYOUT_PLAN: u32 = 4;
const LAYOUT_MAGIC: &[u8; 4] = b"KTL1";
const LAYOUT_ENTRY_SIZE: usize = 24;

const MAGIC: &[u8; 4] = b"KOFF";
const MANIFEST_MAGIC: &[u8; 4] = b"KMF1";

#[derive(Debug)]
pub enum FormatError {
    Invalid(&'static str),
    Overflow,
    UnsupportedVersion(u16, u16),
    Io(std::io::Error),
}
impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid .koff: {s}"),
            Self::Overflow => write!(f, "offset or length overflow"),
            Self::UnsupportedVersion(major, minor) => {
                write!(f, "unsupported .koff version {major}.{minor}")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}
impl std::error::Error for FormatError {}
impl From<std::io::Error> for FormatError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub major: u16,
    pub minor: u16,
    pub flags: u32,
    pub file_size: u64,
    pub directory_offset: u64,
    pub directory_length: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub kind: u32,
    pub flags: u32,
    pub offset: u64,
    pub length: u64,
    pub logical_length: u64,
    pub alignment: u32,
    pub id: u32,
    pub crc32: u32,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub model_name: String,
    pub model_format: u16, // v0.1: 1 = ONNX
    pub backend_hint: u16, // v0.1: 1 = ONNX Runtime
    pub payload_id: u32,
}
impl Manifest {
    pub fn onnx(model_name: impl Into<String>, payload_id: u32) -> Self {
        Self {
            model_name: model_name.into(),
            model_format: 1,
            backend_hint: 1,
            payload_id,
        }
    }
    pub fn encode(&self) -> Result<Vec<u8>, FormatError> {
        let name = self.model_name.as_bytes();
        let len = u32::try_from(name.len()).map_err(|_| FormatError::Overflow)?;
        if name.is_empty() || name.len() > 4096 {
            return Err(FormatError::Invalid("model name length"));
        }
        let mut out = Vec::with_capacity(20 + name.len());
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&self.model_format.to_le_bytes());
        out.extend_from_slice(&self.backend_hint.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&self.payload_id.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(name);
        Ok(out)
    }
    pub fn decode(data: &[u8]) -> Result<Self, FormatError> {
        if data.len() < 20 || &data[..4] != MANIFEST_MAGIC {
            return Err(FormatError::Invalid("manifest magic or size"));
        }
        let len = u32_at(data, 8)? as usize;
        if len == 0 || len > 4096 || data.len() != 20 + len {
            return Err(FormatError::Invalid("manifest name length"));
        }
        if u32_at(data, 16)? != 0 {
            return Err(FormatError::Invalid("manifest reserved bytes"));
        }
        let model_name = std::str::from_utf8(&data[20..])
            .map_err(|_| FormatError::Invalid("manifest UTF-8"))?
            .to_owned();
        Ok(Self {
            model_name,
            model_format: u16_at(data, 4)?,
            backend_hint: u16_at(data, 6)?,
            payload_id: u32_at(data, 12)?,
        })
    }
}

fn u16_at(data: &[u8], at: usize) -> Result<u16, FormatError> {
    Ok(u16::from_le_bytes(
        data.get(at..at + 2)
            .ok_or(FormatError::Invalid("short field"))?
            .try_into()
            .unwrap(),
    ))
}
fn u32_at(data: &[u8], at: usize) -> Result<u32, FormatError> {
    Ok(u32::from_le_bytes(
        data.get(at..at + 4)
            .ok_or(FormatError::Invalid("short field"))?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(data: &[u8], at: usize) -> Result<u64, FormatError> {
    Ok(u64::from_le_bytes(
        data.get(at..at + 8)
            .ok_or(FormatError::Invalid("short field"))?
            .try_into()
            .unwrap(),
    ))
}
fn range(
    offset: u64,
    length: u64,
    file_length: usize,
) -> Result<std::ops::Range<usize>, FormatError> {
    let end = offset.checked_add(length).ok_or(FormatError::Overflow)?;
    if end > file_length as u64 {
        return Err(FormatError::Invalid("section outside file"));
    }
    Ok(usize::try_from(offset).map_err(|_| FormatError::Overflow)?
        ..usize::try_from(end).map_err(|_| FormatError::Overflow)?)
}
fn align_up(n: usize, a: usize) -> Result<usize, FormatError> {
    if a == 0 || !a.is_power_of_two() || a > 1_048_576 {
        return Err(FormatError::Invalid("invalid alignment"));
    }
    n.checked_add(a - 1)
        .map(|x| x & !(a - 1))
        .ok_or(FormatError::Overflow)
}
/// CRC32 IEEE 802.3: integrity against accidental corruption, NOT authenticity.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[derive(Debug)]
pub struct KoffFile {
    data: Vec<u8>,
    pub header: Header,
    pub sections: Vec<Section>,
    pub manifest: Manifest,
}
impl KoffFile {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, FormatError> {
        Self::parse(fs::read(path)?)
    }
    pub fn parse(data: Vec<u8>) -> Result<Self, FormatError> {
        if data.len() < HEADER_SIZE || &data[0..4] != MAGIC {
            return Err(FormatError::Invalid("header magic/size"));
        }
        let major = u16_at(&data, 4)?;
        let minor = u16_at(&data, 6)?;
        if major != 0 || !(minor == 1 || minor == 2) {
            return Err(FormatError::UnsupportedVersion(major, minor));
        }
        if u32_at(&data, 8)? != 0 {
            return Err(FormatError::Invalid("unknown header flags"));
        }
        if u32_at(&data, 12)? != HEADER_SIZE as u32 {
            return Err(FormatError::Invalid("header size"));
        }
        let file_size = u64_at(&data, 16)?;
        if file_size != data.len() as u64 {
            return Err(FormatError::Invalid("file size mismatch"));
        }
        let directory_offset = u64_at(&data, 24)?;
        let directory_length = u64_at(&data, 32)?;
        if data[40..64].iter().any(|&b| b != 0) {
            return Err(FormatError::Invalid("reserved header bytes"));
        }
        if directory_offset < HEADER_SIZE as u64
            || directory_length == 0
            || directory_length % DIRECTORY_ENTRY_SIZE as u64 != 0
        {
            return Err(FormatError::Invalid("directory layout"));
        }
        let count = (directory_length / DIRECTORY_ENTRY_SIZE as u64) as usize;
        if count > MAX_SECTIONS {
            return Err(FormatError::Invalid("too many sections"));
        }
        let dir = range(directory_offset, directory_length, data.len())?;
        let mut sections = Vec::with_capacity(count);
        let mut ranges = vec![
            (0u64, HEADER_SIZE as u64),
            (directory_offset, directory_offset + directory_length),
        ];
        for index in 0..count {
            let p = dir.start + index * DIRECTORY_ENTRY_SIZE;
            let kind = u32_at(&data, p)?;
            let flags = u32_at(&data, p + 4)?;
            let offset = u64_at(&data, p + 8)?;
            let length = u64_at(&data, p + 16)?;
            let logical_length = u64_at(&data, p + 24)?;
            let alignment = u32_at(&data, p + 32)?;
            let id = u32_at(&data, p + 36)?;
            let checksum = u32_at(&data, p + 40)?;
            if u32_at(&data, p + 44)? != 0 {
                return Err(FormatError::Invalid("section reserved bytes"));
            }
            if flags & !CRITICAL != 0 {
                return Err(FormatError::Invalid("unknown section flags"));
            }
            if !alignment.is_power_of_two() || alignment == 0 || alignment > 1_048_576 {
                return Err(FormatError::Invalid("section alignment"));
            }
            if offset % alignment as u64 != 0 || length == 0 || logical_length != length {
                return Err(FormatError::Invalid("section alignment/length"));
            }
            let r = range(offset, length, data.len())?;
            if crc32(&data[r]) != checksum {
                return Err(FormatError::Invalid("section checksum"));
            }
            if ![
                SECTION_MANIFEST,
                SECTION_MODEL_PAYLOAD,
                SECTION_EXTERNAL_WEIGHTS,
                SECTION_LAYOUT_PLAN,
            ]
            .contains(&kind)
                && flags & CRITICAL != 0
            {
                return Err(FormatError::Invalid("unknown required section"));
            }
            ranges.push((offset, offset + length));
            sections.push(Section {
                kind,
                flags,
                offset,
                length,
                logical_length,
                alignment,
                id,
                crc32: checksum,
            });
        }
        ranges.sort_unstable_by_key(|&(start, _)| start);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(FormatError::Invalid("overlapping sections"));
            }
        }
        let manifests: Vec<_> = sections
            .iter()
            .filter(|s| s.kind == SECTION_MANIFEST)
            .collect();
        if manifests.len() != 1 {
            return Err(FormatError::Invalid(
                "exactly one manifest required in v0.1",
            ));
        }
        let mr = range(manifests[0].offset, manifests[0].length, data.len())?;
        let manifest = Manifest::decode(&data[mr])?;
        let payloads: Vec<_> = sections
            .iter()
            .filter(|s| s.kind == SECTION_MODEL_PAYLOAD && s.id == manifest.payload_id)
            .collect();
        if payloads.len() != 1 || manifest.model_format != 1 || manifest.backend_hint != 1 {
            return Err(FormatError::Invalid("unsupported/missing ONNX payload"));
        }
        let external: Vec<_> = sections
            .iter()
            .filter(|s| s.kind == SECTION_EXTERNAL_WEIGHTS)
            .collect();
        let layouts: Vec<_> = sections
            .iter()
            .filter(|s| s.kind == SECTION_LAYOUT_PLAN)
            .collect();
        if external.len() > 1 || layouts.len() > 1 || (external.len() != layouts.len()) {
            return Err(FormatError::Invalid(
                "external weights require exactly one layout plan",
            ));
        }
        if minor == 1 && (!external.is_empty() || !layouts.is_empty()) {
            return Err(FormatError::Invalid("external weights require KOFF 0.2"));
        }
        if let (Some(weight_section), Some(layout_section)) = (external.first(), layouts.first()) {
            let layout_bytes = range(layout_section.offset, layout_section.length, data.len())?;
            parse_layout(&data[layout_bytes], weight_section.length)?;
        }
        let mut seen = std::collections::BTreeSet::new();
        for section in &sections {
            if !seen.insert((section.kind, section.id)) {
                return Err(FormatError::Invalid("duplicate section kind/id"));
            }
        }
        Ok(Self {
            data,
            header: Header {
                major,
                minor,
                flags: 0,
                file_size,
                directory_offset,
                directory_length,
            },
            sections,
            manifest,
        })
    }
    pub fn payload(&self) -> &[u8] {
        let s = self
            .sections
            .iter()
            .find(|s| s.kind == SECTION_MODEL_PAYLOAD && s.id == self.manifest.payload_id)
            .unwrap();
        &self.data[s.offset as usize..(s.offset + s.length) as usize]
    }
    /// External ONNX weights are stored as a contiguous binary section.
    /// Offset values inside the ONNX graph are relative to the start of this section.
    pub fn external_weights(&self) -> Option<&[u8]> {
        self.section_bytes(SECTION_EXTERNAL_WEIGHTS)
    }
    pub fn layout_bytes(&self) -> Option<&[u8]> {
        self.section_bytes(SECTION_LAYOUT_PLAN)
    }
    pub fn section_bytes(&self, kind: u32) -> Option<&[u8]> {
        self.sections
            .iter()
            .find(|s| s.kind == kind)
            .map(|s| &self.data[s.offset as usize..(s.offset + s.length) as usize])
    }
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
}

/// v0.1 packs one opaque ONNX payload; no ONNX parsing, streaming or mmap yet.
pub fn pack_onnx(
    name: &str,
    payload: &[u8],
    payload_alignment: u32,
) -> Result<Vec<u8>, FormatError> {
    if payload.is_empty() {
        return Err(FormatError::Invalid("empty payload"));
    }
    if payload_alignment < 16 {
        return Err(FormatError::Invalid("payload alignment must be >=16"));
    }
    let manifest = Manifest::onnx(name, 1).encode()?;
    let dir_offset = HEADER_SIZE;
    let dir_length = DIRECTORY_ENTRY_SIZE * 2;
    let manifest_offset = align_up(dir_offset + dir_length, 16)?;
    let payload_offset = align_up(
        manifest_offset
            .checked_add(manifest.len())
            .ok_or(FormatError::Overflow)?,
        payload_alignment as usize,
    )?;
    let file_len = payload_offset
        .checked_add(payload.len())
        .ok_or(FormatError::Overflow)?;
    let mut out = vec![0u8; file_len];
    out[0..4].copy_from_slice(MAGIC);
    out[4..6].copy_from_slice(&0u16.to_le_bytes());
    out[6..8].copy_from_slice(&1u16.to_le_bytes());
    out[12..16].copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
    out[16..24].copy_from_slice(&(file_len as u64).to_le_bytes());
    out[24..32].copy_from_slice(&(dir_offset as u64).to_le_bytes());
    out[32..40].copy_from_slice(&(dir_length as u64).to_le_bytes());
    let sections = [
        (
            SECTION_MANIFEST,
            manifest_offset,
            manifest.len(),
            16u32,
            0u32,
            crc32(&manifest),
        ),
        (
            SECTION_MODEL_PAYLOAD,
            payload_offset,
            payload.len(),
            payload_alignment,
            1u32,
            crc32(payload),
        ),
    ];
    for (i, (kind, offset, length, alignment, id, checksum)) in sections.into_iter().enumerate() {
        let p = dir_offset + i * DIRECTORY_ENTRY_SIZE;
        out[p..p + 4].copy_from_slice(&kind.to_le_bytes());
        out[p + 4..p + 8].copy_from_slice(&CRITICAL.to_le_bytes());
        out[p + 8..p + 16].copy_from_slice(&(offset as u64).to_le_bytes());
        out[p + 16..p + 24].copy_from_slice(&(length as u64).to_le_bytes());
        out[p + 24..p + 32].copy_from_slice(&(length as u64).to_le_bytes());
        out[p + 32..p + 36].copy_from_slice(&alignment.to_le_bytes());
        out[p + 36..p + 40].copy_from_slice(&id.to_le_bytes());
        out[p + 40..p + 44].copy_from_slice(&checksum.to_le_bytes());
    }
    out[manifest_offset..manifest_offset + manifest.len()].copy_from_slice(&manifest);
    out[payload_offset..].copy_from_slice(payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
    #[test]
    fn roundtrip() {
        let payload = b"not-a-real-onnx-file";
        let bytes = pack_onnx("sample", payload, 4096).unwrap();
        let parsed = KoffFile::parse(bytes).unwrap();
        assert_eq!(parsed.manifest.model_name, "sample");
        assert_eq!(parsed.payload(), payload);
        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.sections[1].offset % 4096, 0);
    }
    #[test]
    fn detects_corruption() {
        let mut bytes = pack_onnx("sample", b"model-blob", 64).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        assert!(KoffFile::parse(bytes).is_err());
    }
    #[test]
    fn detects_bad_offset() {
        let mut bytes = pack_onnx("sample", b"payload", 64).unwrap();
        let p = HEADER_SIZE + DIRECTORY_ENTRY_SIZE + 8;
        bytes[p..p + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(KoffFile::parse(bytes).is_err());
    }
    #[test]
    fn rejects_bad_alignment() {
        assert!(pack_onnx("sample", b"data", 31).is_err());
    }
}

/// One tensor's position in a packed external-weights blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlacement {
    pub tensor_id: u32,
    pub first_use_step: u32,
    pub offset: u64,
    pub length: u64,
}

/// A deliberately small, typed binary format; no JSON is stored in a .koff file.
/// Wire layout: KTL1, alignment:u32, count:u32, reserved:u32, then 24-byte records.
pub fn encode_layout(entries: &[TensorPlacement], alignment: u32) -> Result<Vec<u8>, FormatError> {
    if !(16..=1_048_576).contains(&alignment) || !alignment.is_power_of_two() {
        return Err(FormatError::Invalid(
            "layout alignment must be a power of two >= 16",
        ));
    }
    if entries.is_empty() || entries.len() > 1_000_000 {
        return Err(FormatError::Invalid("invalid number of layout entries"));
    }
    let count = u32::try_from(entries.len()).map_err(|_| FormatError::Overflow)?;
    let capacity = 16usize
        .checked_add(
            entries
                .len()
                .checked_mul(LAYOUT_ENTRY_SIZE)
                .ok_or(FormatError::Overflow)?,
        )
        .ok_or(FormatError::Overflow)?;
    let mut result = Vec::with_capacity(capacity);
    result.extend_from_slice(LAYOUT_MAGIC);
    result.extend_from_slice(&alignment.to_le_bytes());
    result.extend_from_slice(&count.to_le_bytes());
    result.extend_from_slice(&0u32.to_le_bytes());
    for e in entries {
        result.extend_from_slice(&e.tensor_id.to_le_bytes());
        result.extend_from_slice(&e.first_use_step.to_le_bytes());
        result.extend_from_slice(&e.offset.to_le_bytes());
        result.extend_from_slice(&e.length.to_le_bytes());
    }
    Ok(result)
}

pub fn parse_layout(
    bytes: &[u8],
    weight_blob_length: u64,
) -> Result<Vec<TensorPlacement>, FormatError> {
    if bytes.len() < 16 || &bytes[..4] != LAYOUT_MAGIC || u32_at(bytes, 12)? != 0 {
        return Err(FormatError::Invalid("layout header"));
    }
    let alignment = u32_at(bytes, 4)?;
    let count = u32_at(bytes, 8)? as usize;
    if !(16..=1_048_576).contains(&alignment)
        || !alignment.is_power_of_two()
        || count == 0
        || count > 1_000_000
    {
        return Err(FormatError::Invalid("layout alignment/count"));
    }
    let expected = count
        .checked_mul(LAYOUT_ENTRY_SIZE)
        .and_then(|n| n.checked_add(16))
        .ok_or(FormatError::Overflow)?;
    if bytes.len() != expected {
        return Err(FormatError::Invalid("layout byte length"));
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut ranges = Vec::with_capacity(count);
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let at = 16 + i * LAYOUT_ENTRY_SIZE;
        let tensor_id = u32_at(bytes, at)?;
        let first_use_step = u32_at(bytes, at + 4)?;
        let offset = u64_at(bytes, at + 8)?;
        let length = u64_at(bytes, at + 16)?;
        if !ids.insert(tensor_id) || length == 0 || offset % alignment as u64 != 0 {
            return Err(FormatError::Invalid(
                "duplicate id or invalid tensor placement",
            ));
        }
        let end = offset.checked_add(length).ok_or(FormatError::Overflow)?;
        if end > weight_blob_length {
            return Err(FormatError::Invalid("tensor outside weights blob"));
        }
        ranges.push((offset, end));
        entries.push(TensorPlacement {
            tensor_id,
            first_use_step,
            offset,
            length,
        });
    }
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(FormatError::Invalid("overlapping weight ranges"));
        }
    }
    Ok(entries)
}

/// Builds an actual KOFF v0.2 file containing an ONNX graph, packed external weights,
/// and a topology-aware index. It does NOT rewrite ONNX external_data references:
/// the caller must supply a graph whose offsets agree with `layout`.
pub fn pack_onnx_external(
    name: &str,
    onnx_graph: &[u8],
    packed_weights: &[u8],
    layout: &[u8],
    weight_alignment: u32,
) -> Result<Vec<u8>, FormatError> {
    if onnx_graph.is_empty() || packed_weights.is_empty() {
        return Err(FormatError::Invalid("empty ONNX graph or weights"));
    }
    if weight_alignment < 16 {
        return Err(FormatError::Invalid("weight alignment must be >=16"));
    }
    let placements = parse_layout(layout, packed_weights.len() as u64)?;
    if placements.is_empty() {
        return Err(FormatError::Invalid("empty layout"));
    }
    let recorded_alignment = u32_at(layout, 4)?;
    if recorded_alignment != weight_alignment {
        return Err(FormatError::Invalid("alignment mismatch"));
    }
    let manifest = Manifest::onnx(name, 1).encode()?;
    let inputs: [(u32, &[u8], u32, u32); 4] = [
        (SECTION_MANIFEST, &manifest, 16, 0),
        (SECTION_MODEL_PAYLOAD, onnx_graph, 64, 1),
        (SECTION_LAYOUT_PLAN, layout, 16, 0),
        (
            SECTION_EXTERNAL_WEIGHTS,
            packed_weights,
            weight_alignment,
            1,
        ),
    ];
    let directory_offset = HEADER_SIZE;
    let directory_length = inputs.len() * DIRECTORY_ENTRY_SIZE;
    let mut cursor = directory_offset + directory_length;
    let mut sections = Vec::with_capacity(inputs.len());
    for (kind, data, alignment, id) in &inputs {
        let offset = align_up(cursor, *alignment as usize)?;
        cursor = offset
            .checked_add(data.len())
            .ok_or(FormatError::Overflow)?;
        sections.push((*kind, offset, data.len(), *alignment, *id, crc32(data)));
    }
    let mut out = vec![0u8; cursor];
    out[..4].copy_from_slice(MAGIC);
    out[4..6].copy_from_slice(&0u16.to_le_bytes());
    out[6..8].copy_from_slice(&2u16.to_le_bytes());
    out[12..16].copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
    out[16..24].copy_from_slice(&(cursor as u64).to_le_bytes());
    out[24..32].copy_from_slice(&(directory_offset as u64).to_le_bytes());
    out[32..40].copy_from_slice(&(directory_length as u64).to_le_bytes());
    for (i, (kind, offset, length, alignment, id, checksum)) in sections.iter().enumerate() {
        let p = directory_offset + i * DIRECTORY_ENTRY_SIZE;
        out[p..p + 4].copy_from_slice(&kind.to_le_bytes());
        out[p + 4..p + 8].copy_from_slice(&CRITICAL.to_le_bytes());
        out[p + 8..p + 16].copy_from_slice(&(*offset as u64).to_le_bytes());
        out[p + 16..p + 24].copy_from_slice(&(*length as u64).to_le_bytes());
        out[p + 24..p + 32].copy_from_slice(&(*length as u64).to_le_bytes());
        out[p + 32..p + 36].copy_from_slice(&alignment.to_le_bytes());
        out[p + 36..p + 40].copy_from_slice(&id.to_le_bytes());
        out[p + 40..p + 44].copy_from_slice(&checksum.to_le_bytes());
        out[*offset..*offset + *length].copy_from_slice(inputs[i].1);
    }
    Ok(out)
}

#[cfg(test)]
mod topology_tests {
    use super::*;
    #[test]
    fn external_weights_round_trip() {
        let plan = encode_layout(
            &[
                TensorPlacement {
                    tensor_id: 7,
                    first_use_step: 0,
                    offset: 0,
                    length: 3,
                },
                TensorPlacement {
                    tensor_id: 2,
                    first_use_step: 1,
                    offset: 64,
                    length: 4,
                },
            ],
            64,
        )
        .unwrap();
        let mut weights = vec![0u8; 68];
        weights[..3].copy_from_slice(b"abc");
        weights[64..].copy_from_slice(b"defg");
        let data = pack_onnx_external("demo", b"onnx graph bytes", &weights, &plan, 64).unwrap();
        let parsed = KoffFile::parse(data).unwrap();
        assert_eq!(parsed.header.minor, 2);
        assert_eq!(parsed.external_weights().unwrap(), weights);
        assert_eq!(
            parse_layout(parsed.layout_bytes().unwrap(), 68)
                .unwrap()
                .len(),
            2
        );
    }
    #[test]
    fn rejects_bad_external_layout() {
        let p = encode_layout(
            &[TensorPlacement {
                tensor_id: 1,
                first_use_step: 0,
                offset: 128,
                length: 4,
            }],
            64,
        )
        .unwrap();
        assert!(pack_onnx_external("bad", b"graph", b"1234", &p, 64).is_err());
    }
    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let p = encode_layout(
            &[
                TensorPlacement {
                    tensor_id: 1,
                    first_use_step: 0,
                    offset: 0,
                    length: 70,
                },
                TensorPlacement {
                    tensor_id: 2,
                    first_use_step: 1,
                    offset: 64,
                    length: 10,
                },
            ],
            64,
        )
        .unwrap();
        assert!(parse_layout(&p, 74).is_err());
    }
}
