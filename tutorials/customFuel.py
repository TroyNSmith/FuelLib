import fuellib as fl

# Load an embedded fuel
fuel = fl.Fuel.from_name("posf10264")

print(f"Fuel: {fuel.name}")
print(f"Number of compounds: {fuel.num_compounds}")

# To use a custom fuel, create a directory containing one "<name>.json" file per
# fuel (see fuellib/data/fuel/decane.json for the expected schema):
#
# customFuels/fuelData/
#   └── hefa-S1.json
#
# Then load it with:
try:
    custom_fuel = fl.Fuel.from_name("hefa-S1", fuel_data_dir="customFuels/fuelData")

    print(f"\nFuel: {custom_fuel.name}")
    print(f"Number of compounds: {custom_fuel.num_compounds}")
except FileNotFoundError:
    print(
        "\n(Skipping custom fuel example: 'customFuels/fuelData/hefa-S1.json' not present)"
    )
