import numpy as np
import pandas as pd
import os
import json


from sklearn.preprocessing import LabelEncoder



# int64 is not JSON Serializable
# Use this class to generate the json metadata
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):  # Convert all NumPy integers

            return int(obj)

        return super().default(obj)
    

def main():
    directories = [
        "data/per_store",
        "data/per_drink", 
        "data/per_city",
        "data/panels",
        "data/panels/store",
        "data/panels/drink"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Optional: print confirmation
        print(f"Ensured exists: {directory}")
    try: 
        sgor = pd.read_csv("data/selangor.csv", index_col=False) # sgor is the raw dataset
        
    except FileNotFoundError:
        print(f"Download selangor.csv and place in a folder called \"data\" in {os.getcwd()}")
        
    
    sgor.drop(sgor.loc[sgor["Net.Selling.Price"] < 0].index, inplace = True)
    sgor.drop(sgor.loc[sgor["City"] == "Kuala Lumpur"].index, inplace = True)
    
    sgor.loc[sgor["Product.Name"] == "DARK CHOCO LATTE", "Product.Name"] = "DARK CHOCOLATE LATTE"

    sgor_clean = sgor.copy() 
    
    cat_features = ["Store.Code", "City", "State", "Type", "Venue", "Payment.Method", "Product.Code", "Product.Name", "Temperature", "Sub.Category"]
    
    encodings_dict = dict(zip(cat_features, np.zeros(len(cat_features))))

    # Label-encoding steps
    for feature in cat_features:
        label_encoder = LabelEncoder()
        
        encoded_feature = label_encoder.fit_transform(sgor_clean[feature])

        encodings_dict[feature] = dict(zip([str(x) for x in label_encoder.transform(label_encoder.classes_)], label_encoder.classes_))
        
        sgor_clean[feature] = encoded_feature
        
        


        
    # This section should be pretty self-explanatory
    na_bean_mask = pd.isna(sgor_clean["Bean"])
    na_milk_mask = pd.isna(sgor_clean["Milk"])
    
    sgor_clean.loc[na_bean_mask, "Bean"] = "X"
    sgor_clean.loc[na_milk_mask, "Milk"] = "X"

    
    # Fit LabelEncoders
    bean_label_encoder = LabelEncoder().fit(sgor_clean["Bean"])
    milk_label_encoder = LabelEncoder().fit(sgor_clean["Milk"])
    
    # Save the encodings to the dictionary
    encodings_dict["Bean"] = dict(zip([str(x) for x in bean_label_encoder.transform(bean_label_encoder.classes_)], bean_label_encoder.classes_))
    encodings_dict["Milk"] = dict(zip([str(x) for x in milk_label_encoder.transform(milk_label_encoder.classes_)], milk_label_encoder.classes_))   
    
    sgor_clean["Bean"] = bean_label_encoder.transform(sgor_clean["Bean"])
    sgor_clean["Milk"] = milk_label_encoder.transform(sgor_clean["Milk"])
    with open("data/metadata.json", "w") as file:
        json.dump(encodings_dict, file, indent = 4)
    

    sgor_clean.drop(columns=["ID"], inplace = True)
    
    print("sgor_clean DataFrame produced. Saving csv...")
    sgor_clean.to_csv("data/sgor_clean.csv", index = False)
    assert os.path.exists("data/sgor_clean.csv"), "data/sgor_clean.csv FAILED TO SAVE"
    
    
    
    revenue_per_store = sgor_clean[["Sales.Date", "Store.Code", "Net.Selling.Price"]]
    revenue_per_store = revenue_per_store.pivot_table(index = "Sales.Date", columns = "Store.Code", values="Net.Selling.Price", aggfunc="sum").reset_index()
    
    revenue_per_store["Sales.Date"] = pd.to_datetime(
        revenue_per_store["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )    
    
    discount_per_store = sgor_clean[["Sales.Date", "Store.Code", "Discount"]]
    discount_per_store = discount_per_store.pivot_table(index = "Sales.Date", columns = "Store.Code", values="Discount", aggfunc="sum").reset_index()
    
    discount_per_store["Sales.Date"] = pd.to_datetime(
        discount_per_store["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )    

    all_store_codes = list(encodings_dict["Store.Code"].keys())

    for store_code in all_store_codes:
        store_code_data = sgor_clean.loc[sgor_clean["Store.Code"] == int(store_code)]
        
        store_code_open_date = store_code_data["Opening.Date"].iloc[0]
        
        store_code_open_date = pd.to_datetime(
            store_code_data["Opening.Date"].iloc[0],
            format="%d/%m/%Y"
        )
        
        
        no_sale_mask = (discount_per_store["Sales.Date"] >= store_code_open_date) & (pd.isna(revenue_per_store[int(store_code)]))  

        discount_per_store.loc[no_sale_mask, int(store_code)] = 0
        revenue_per_store.loc[no_sale_mask, int(store_code)] = 0

    revenue_per_store["Total"] = np.apply_along_axis(arr = np.array(revenue_per_store.drop(columns = "Sales.Date", inplace = False)), axis = 1, func1d=np.nansum)
    discount_per_store["Total"] = np.apply_along_axis(arr = np.array(discount_per_store.drop(columns = "Sales.Date", inplace = False)), axis = 1, func1d=np.nansum)
    
    print("revenue_per_store DataFrame cleaned. Saving csv...")
    revenue_per_store.to_csv("data/per_store/revenue_per_store.csv", index = False)
    assert os.path.exists("data/per_store/revenue_per_store.csv"), "data/per_store/revenue_per_store.csv FAILED TO SAVE"
    

    print("discount_per_store DataFrame cleaned. Saving csv...")
    discount_per_store.to_csv("data/per_store/discount_per_store.csv", index = False)
    assert os.path.exists("data/per_store/discount_per_store.csv"), "data/per_store/discount_per_store.csv FAILED TO SAVE"
    

    
    
    all_store_codes = np.unique(revenue_per_store.drop(columns = ["Sales.Date", "Total"]).columns)
  

    
    revenue_per_city = sgor_clean[["Sales.Date", "City", "Net.Selling.Price"]]
    revenue_per_city = revenue_per_city.pivot_table(index = "Sales.Date", columns = "City", values="Net.Selling.Price", aggfunc="sum").reset_index()
    
    revenue_per_city["Sales.Date"] = pd.to_datetime(
        revenue_per_city["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )    
    

    all_cities = list(encodings_dict["City"].keys())

    for city in all_cities:
        city_data = sgor_clean.loc[sgor_clean["Store.Code"] == int(city)]
        
        city_active_date = city_data["Opening.Date"].iloc[0]

        no_sale_mask = (revenue_per_city["Sales.Date"] >= city_active_date) & (pd.isna(revenue_per_city[int(city)]))    
        revenue_per_city.loc[no_sale_mask, int(city)] = 0


    revenue_per_city["Total"] = np.apply_along_axis(arr = np.array(revenue_per_city.drop(columns = "Sales.Date", inplace = False)), axis = 1, func1d=np.nansum)

    print("revenue_per_city DataFrame cleaned. Saving csv...")
    revenue_per_city.to_csv("data/per_city/revenue_per_city.csv", index = False)
    assert os.path.exists("data/per_city/revenue_per_city.csv"), "data/per_city/revenue_per_city.csv FAILED TO SAVE"    
    
    
    revenue_per_drink = sgor_clean[["Sales.Date", "Product.Name", "Net.Selling.Price"]]
    revenue_per_drink = revenue_per_drink.pivot_table(index = "Sales.Date", columns = "Product.Name", values="Net.Selling.Price", aggfunc="sum").reset_index()
    revenue_per_drink = revenue_per_drink.fillna(value = 0)
    revenue_per_drink["Total"] = np.apply_along_axis(arr = np.array(revenue_per_drink.drop(columns = "Sales.Date", inplace = False)), axis = 1, func1d=np.nansum)

    print("revenue_per_drink DataFrame produced. Saving csv...")
    revenue_per_drink.to_csv("data/per_drink/revenue_per_drink.csv", index = False)
    assert os.path.exists("data/per_drink/revenue_per_drink.csv"), "data/per_drink/revenue_per_drink.csv FAILED TO SAVE"
    
    discount_per_drink = sgor_clean[["Sales.Date", "Product.Name", "Discount"]]
    discount_per_drink = discount_per_drink.pivot_table(index = "Sales.Date", columns = "Product.Name", values="Discount", aggfunc="sum").reset_index()
    discount_per_drink = discount_per_drink.fillna(value = 0)
    discount_per_drink["Total"] = np.apply_along_axis(arr = np.array(discount_per_drink.drop(columns = "Sales.Date", inplace = False)), axis = 1, func1d=np.nansum)

    print("discount_per_drink DataFrame produced. Saving csv...")
    discount_per_drink.to_csv("data/per_drink/discount_per_drink.csv", index = False)
    assert os.path.exists("data/per_drink/discount_per_drink.csv"), "data/per_drink/discount_per_drink.csv FAILED TO SAVE"
    
    
    store_to_city = {}
    for city in sgor_clean["City"].unique():
        stores_in_city = sgor_clean.loc[sgor_clean["City"] == city, "Store.Code"].unique()
        for store in stores_in_city:
            store_to_city[store] = city
        print(f"City: {city}, Stores: {stores_in_city}")
        
    drink_to_subcat = {}
    for subcat in sgor_clean["Sub.Category"].unique():
        drinks_in_subcat = sgor_clean.loc[sgor_clean["Sub.Category"] == subcat, "Product.Name"].unique()
        for drink in drinks_in_subcat:
            drink_to_subcat[drink] = subcat
        print(f"Sub.Category: {subcat}, Drinks : {drinks_in_subcat}")
        
    store_to_city = dict(sorted(store_to_city.items(), key = lambda item: item[0]))

    store_to_city = {int(k): v for k, v in store_to_city.items()}

    # 1. Melt the Discount data
    discount_store_panel = discount_per_store.melt(
        id_vars='Sales.Date', 
        var_name='Store.Code', 
        value_name='Discount'
    ).dropna()

    # 2. Melt the Revenue data
    revenue_store_panel = revenue_per_store.drop(columns=['Total']).melt(
        id_vars='Sales.Date',
        var_name='Store.Code',
        value_name='Revenue'
    ).dropna()


    

    # 1. Melt the Discount data
    discount_drink_panel = discount_per_drink.melt(
        id_vars='Sales.Date', 
        var_name='Product.Name', 
        value_name='Discount'
    ).dropna()

    # 2. Melt the Revenue data
    revenue_drink_panel = revenue_per_drink.drop(columns=['Total']).melt(
        id_vars='Sales.Date',
        var_name='Product.Name',
        value_name='Revenue'
    ).dropna()

    # 3. Combine the data 
    # Note: Using merge is safer than direct assignment to ensure rows align by Date and Store
    combined_store_panel = pd.merge(revenue_store_panel, discount_store_panel, on=['Sales.Date', 'Store.Code'])
    combined_store_panel["City"] = combined_store_panel["Store.Code"].map(store_to_city)
    
    combined_drink_panel = pd.merge(revenue_drink_panel, discount_drink_panel, on=['Sales.Date', 'Product.Name'])
    combined_drink_panel["Sub.Category"] = combined_drink_panel["Product.Name"].map(drink_to_subcat)
    

    revenue_store_panel["City"] = revenue_store_panel["Store.Code"].map(store_to_city)
    discount_store_panel["City"] = discount_store_panel["Store.Code"].map(store_to_city)
    
    revenue_drink_panel["Sub.Category"] = revenue_drink_panel["Product.Name"].map(drink_to_subcat)
    discount_drink_panel["Sub.Category"] = discount_drink_panel["Product.Name"].map(drink_to_subcat)
    
    
    
    
    
    
    # 5. Save all dataframes to the 'data' folder
    output_dir = 'data/panels'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the files to save
    files = {
        "store/discount_store_panel.csv": discount_store_panel,
        "store/revenue_store_panel.csv": revenue_store_panel,
        "store/combined_store_panel.csv": combined_store_panel,
        
        "drink/discount_drink_panel.csv": discount_drink_panel,
        "drink/revenue_drink_panel.csv": revenue_drink_panel,        
        "drink/combined_drink_panel.csv": combined_drink_panel
    }

    for filename, df in files.items():
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")
        
        



  

if __name__ == "__main__":
    main()
    
    sgor_clean = pd.read_csv("data/sgor_clean.csv", index_col = False)
    print(sgor_clean.head())
    
    revenue_per_store = pd.read_csv("data/per_store/revenue_per_store.csv", index_col = False)
    print(revenue_per_store.head())
    
    revenue_per_city = pd.read_csv("data/per_city/revenue_per_city.csv", index_col = False)
    print(revenue_per_city.head())
    
    revenue_per_drink = pd.read_csv("data/per_drink/revenue_per_drink.csv", index_col = False)
    print(revenue_per_drink.head())
    