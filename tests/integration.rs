mod common;

use libtest_mimic::{Arguments, Failed, Trial};

use self::common::{load_suite, run_test_entry, suites, test_threads, too_big_to_test};

const SUITES: &[&str] = &["duckdb", "fixtures", "kaggle", "postgres", "rust-csv"];

fn main() {
    let mut args = Arguments::from_args();

    if args.test_threads.is_none() {
        args.test_threads = Some(test_threads());
    }

    let tests: Vec<Trial> = suites(SUITES, args.filter.as_deref())
        .into_iter()
        .flat_map(|suite| {
            load_suite(&suite).into_iter().map(move |entry| {
                let name = format!("{}::{}", suite.name, entry.rel_path);
                let ignored = entry.skip || !entry.abs_path.exists() || too_big_to_test(&entry);
                Trial::test(name, move || run_test_entry(&entry).map_err(Failed::from))
                    .with_ignored_flag(ignored)
            })
        })
        .collect();

    libtest_mimic::run(&args, tests).exit();
}
