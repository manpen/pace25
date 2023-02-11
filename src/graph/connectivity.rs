use super::*;

pub trait Connectivity {
    fn partition_into_connected_components(&self, skip_trivial: bool) -> Partition;
}

impl<G> Connectivity for G
where
    G: AdjacencyList,
{
    fn partition_into_connected_components(&self, skip_trivial: bool) -> Partition {
        let mut partition = Partition::new(self.number_of_nodes());

        let start_node = if skip_trivial {
            self.vertices().find(|&u| self.degree_of(u) > 0).unwrap()
        } else {
            0
        };

        let mut bfs = self.bfs(start_node);

        if skip_trivial {
            bfs.exclude_nodes(self.vertices().filter(|&u| self.degree_of(u) == 0));
        }

        loop {
            let class = partition.add_class([]);

            for u in bfs.by_ref() {
                partition.move_node(u, class);
            }

            if !bfs.try_restart_at_unvisited() {
                break;
            }
        }

        partition
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn partition_into_connected_components() {
        let mut graph = AdjArray::new(7);
        graph.add_edges([(1, 2), (2, 3), (4, 5)], EdgeColor::Black);

        {
            let part = graph.partition_into_connected_components(true);
            assert_eq!(part.number_of_classes(), 2);
            assert_eq!(part.number_of_unassigned(), 2);

            for u in graph.vertices() {
                println!("{u} in {:?}", part.class_of_node(u));
            }

            assert_eq!(part.class_of_node(1), part.class_of_node(2));
            assert_eq!(part.class_of_node(1), part.class_of_node(3));
            assert_eq!(part.class_of_node(4), part.class_of_node(5));
            assert_ne!(part.class_of_node(1), part.class_of_node(5));
            assert!(part.class_of_node(0).is_none());
            assert!(part.class_of_node(6).is_none());
        }

        {
            let part = graph.partition_into_connected_components(false);
            assert_eq!(part.number_of_classes(), 4);
            assert_eq!(part.number_of_unassigned(), 0);
            assert!(part.class_of_node(0).is_some());
            assert!(part.class_of_node(6).is_some());
        }
    }
}
