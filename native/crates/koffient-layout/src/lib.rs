//! Graph-aware tensor layout *planner*, independent of ONNX parsing and inference backends.
//! The resulting offsets are proposals: ONNX Runtime does not automatically honor them.
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct Node {
    pub id: u32,
    pub parents: Vec<u32>,
    pub weights: Vec<u32>,
}
#[derive(Debug, Clone)]
pub struct Weight {
    pub id: u32,
    pub size: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub weight_id: u32,
    pub offset: u64,
    pub size: u64,
    pub first_use_step: usize,
}
#[derive(Debug, Clone)]
pub struct LayoutPlan {
    pub execution_order: Vec<u32>,
    pub placements: Vec<Placement>,
    pub total_bytes: u64,
}

fn align_up(n: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() || alignment > 1_048_576 {
        return Err("invalid alignment".into());
    }
    n.checked_add(alignment - 1)
        .map(|x| x & !(alignment - 1))
        .ok_or_else(|| "offset overflow".into())
}

/// Deterministic topological sort; group weights in order of first consumer.
/// Shares one stored copy for weights used by multiple nodes.
pub fn plan(nodes: &[Node], weights: &[Weight], alignment: u64) -> Result<LayoutPlan, String> {
    let mut node_map = BTreeMap::new();
    for node in nodes {
        if node_map.insert(node.id, node).is_some() {
            return Err("duplicate node id".into());
        }
    }
    let mut indegree = BTreeMap::<u32, usize>::new();
    let mut children = BTreeMap::<u32, Vec<u32>>::new();
    for node in nodes {
        indegree.insert(node.id, node.parents.len());
        let unique: BTreeSet<u32> = node.parents.iter().copied().collect();
        if unique.len() != node.parents.len() {
            return Err("duplicate dependency".into());
        }
        for &parent in &node.parents {
            if !node_map.contains_key(&parent) {
                return Err("unknown parent".into());
            }
            children.entry(parent).or_default().push(node.id);
        }
    }
    let mut ready: BTreeSet<_> = indegree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();
    let mut order = vec![];
    while let Some(&id) = ready.iter().next() {
        ready.remove(&id);
        order.push(id);
        for &child in children.get(&id).map(Vec::as_slice).unwrap_or(&[]) {
            let d = indegree.get_mut(&child).ok_or("node missing")?;
            *d -= 1;
            if *d == 0 {
                ready.insert(child);
            }
        }
    }
    if order.len() != nodes.len() {
        return Err("cycle in computation graph".into());
    }
    let mut weight_map = BTreeMap::new();
    for weight in weights {
        if weight.size == 0 || weight_map.insert(weight.id, weight.size).is_some() {
            return Err("invalid or duplicate weight".into());
        }
    }
    let mut first_use = BTreeMap::<u32, usize>::new();
    for (step, id) in order.iter().enumerate() {
        for &weight_id in &node_map[id].weights {
            if !weight_map.contains_key(&weight_id) {
                return Err("unknown weight id".into());
            }
            first_use.entry(weight_id).or_insert(step);
        }
    }
    let mut sorted = Vec::new();
    for (&id, &size) in &weight_map {
        sorted.push((first_use.get(&id).copied().unwrap_or(usize::MAX), id, size));
    }
    sorted.sort_unstable();
    let mut offset = 0u64;
    let mut placements = vec![];
    for (first_use_step, weight_id, size) in sorted {
        offset = align_up(offset, alignment)?;
        placements.push(Placement {
            weight_id,
            offset,
            size,
            first_use_step,
        });
        offset = offset.checked_add(size).ok_or("offset overflow")?;
    }
    Ok(LayoutPlan {
        execution_order: order,
        placements,
        total_bytes: offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn respects_topology_and_shared_weights() {
        let nodes = vec![
            Node {
                id: 3,
                parents: vec![2],
                weights: vec![100],
            },
            Node {
                id: 1,
                parents: vec![],
                weights: vec![200],
            },
            Node {
                id: 2,
                parents: vec![1],
                weights: vec![100, 200],
            },
        ];
        let weights = vec![Weight { id: 100, size: 13 }, Weight { id: 200, size: 7 }];
        let p = plan(&nodes, &weights, 64).unwrap();
        assert_eq!(p.execution_order, vec![1, 2, 3]);
        assert_eq!(p.placements[0].weight_id, 200);
        assert_eq!(p.placements[1].weight_id, 100);
        assert_eq!(p.placements[1].offset, 64);
    }
    #[test]
    fn rejects_cycles() {
        let nodes = vec![
            Node {
                id: 1,
                parents: vec![2],
                weights: vec![],
            },
            Node {
                id: 2,
                parents: vec![1],
                weights: vec![],
            },
        ];
        assert!(plan(&nodes, &[], 64).is_err());
    }
}
