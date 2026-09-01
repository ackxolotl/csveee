use csveee::Parser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = Parser::new();

    let cities = parser.parse(
        "suites/fixtures/standard.csv",
        Vec::new,
        |s, [_name, _age, city]| {
            s.push(city.to_string());
            Ok(())
        },
        |ss| ss.concat(),
    )?;

    println!("The cities: {cities:?}");

    Ok(())
}
