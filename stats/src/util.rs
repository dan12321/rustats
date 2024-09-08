use std::cmp::Ordering;

pub fn sorted_insert<T: Ord>(sorted_vec: &mut Vec<T>, value: T) {
    let mut min = 0;
    let mut max = sorted_vec.len();
    let mut middle = (max + min) / 2;
    while min <= middle && middle < max {
        match value.cmp(&sorted_vec[middle]) {
            Ordering::Less => {
                max = middle;
            },
            Ordering::Greater => {
                min = middle + 1;
            },
            Ordering::Equal => {
                sorted_vec.insert(middle, value);
                return;
            }
        }
        middle = (max + min) / 2;
    }
    sorted_vec.insert(middle, value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_insert_empty() {
        let mut vector = vec![];
        let value = "test";

        sorted_insert(&mut vector, value);

        let expected_value = vec![value];
        assert_eq!(vector, expected_value);
    }

    #[test]
    fn test_sorted_insert_start() {
        let a = "aloft";
        let c = "cast";
        let c2 = "crusty";
        let mut vector = vec![a, c, c2];
        let value = "a";

        sorted_insert(&mut vector, value);

        let expected_value = vec![value, a, c, c2];
        assert_eq!(vector, expected_value);
    }

    #[test]
    fn test_sorted_insert_end() {
        let a = "aloft";
        let c = "cast";
        let c2 = "crusty";
        let d = "deleted";
        let mut vector = vec![a, c, c2, d];
        let value = "test";

        sorted_insert(&mut vector, value);

        let expected_value = vec![a, c, c2, d, value];
        assert_eq!(vector, expected_value);
    }

    #[test]
    fn test_sorted_insert_middle() {
        let a = "aloft";
        let c = "cast";
        let c2 = "crusty";
        let d = "deleted";
        let mut vector = vec![a, c, c2, d];
        let value = "b";

        sorted_insert(&mut vector, value);

        let expected_value = vec![a, value, c, c2, d];
        assert_eq!(vector, expected_value);
    }

    #[test]
    fn test_sorted_insert_equal() {
        let a = "aloft";
        let c = "cast";
        let c2 = "crusty";
        let d = "deleted";
        let e = "elate";
        let mut vector = vec![a, c, c2, d, e];
        let value = "deleted";

        sorted_insert(&mut vector, value);

        let expected_value = vec![a, c, c2, d, value, e];
        assert_eq!(vector, expected_value);
    }
}
