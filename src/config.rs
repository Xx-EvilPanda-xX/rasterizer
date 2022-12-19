use std::str::FromStr;
use core::fmt::Debug;
use super::{Point, Triangle};

pub struct Config {
    pub width: u32,
    pub height: u32,
    pub triangles: Vec<Triangle>,
}

impl Config {
    pub fn new(text: &str) -> Self {
        let mut config = String::new();
        for c in text.chars() {
            if c != ' ' && c != '\r' {
                config.push(c);
            }
        }

        let mut config_processed = String::new();
        for line in config.lines() {
            if line.starts_with("//") {
                continue;
            } else {
                config_processed.push_str(line);
                config_processed.push('\n');
            }
        }

        let mut sections = config_processed.split_inclusive("\n\n");
        let mut img_config = sections.next().expect("missing image config");

        let width = field("width", &mut img_config);
        let height = field("height", &mut img_config);

        let mut triangles = Vec::new();

        for section in sections {
            triangles.push(Triangle::from_config(section));
        }

        Self {
            width: width.parse().expect("Failed to parse width"),
            height: height.parse().expect("Failed to parse height"),
            triangles,
        }
    }
}

impl Triangle {
    pub fn from_config(mut config: &str) -> Self {
        let a = field("a", &mut config);
        let b = field("b", &mut config);
        let c = field("c", &mut config);
        let color_a = field("color_a", &mut config);
        let color_b = field("color_b", &mut config);
        let color_c = field("color_c", &mut config);

        Self {
            a: Point::from_arr(parse_arr(a)),
            b: Point::from_arr(parse_arr(b)),
            c: Point::from_arr(parse_arr(c)),
            color_a: parse_arr(color_a),
            color_b: parse_arr(color_b),
            color_c: parse_arr(color_c),
        }
    }
}

fn parse_arr<T: Copy + Default + FromStr, const N: usize>(mut s: &str) -> [T; N]
        where <T as FromStr>::Err: Debug
{
    s = s.strip_prefix('[').expect("arrays start with `[`");
    s = s.strip_suffix(']').expect("arrays end with `[`");

    let mut elems = s.split(',');
    let mut out = [T::default(); N];
    for i in 0..N {
        let e = elems.next().expect("not enough array elements");
        out[i] = e.parse().expect("Failed to parse array element");
    }

    if elems.next().is_some() {
        panic!("too many array elements");
    }

    out
}

fn field<'a>(name: &str, args: &mut &'a str) -> &'a str {
    *args = args.strip_prefix(&format!("{}=", name)).expect(&format!("missing {}", name));
    let newline = args.find('\n').expect("missing newline");
    let f = &args[..newline];
    *args = &args[newline + 1..];
    f
}