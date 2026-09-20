import numpy as np
import pandas as pd
import json
import calendar
import datetime

DAYS_OF_WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

ORIGIN = datetime.datetime(2023, 1, 1)

REV_BP_DATES = [
    "2023-03-22",
    "2023-06-05",
    "2023-09-27",
    "2024-03-11",
    "2024-06-03"
]

REV_BP_DATES = pd.to_datetime(
    REV_BP_DATES, 
    format="%Y-%m-%d",
    errors = "raise"
)

DISC_BP_DATES = [
    "2023-04-06",
    "2023-06-18",
    "2023-09-06",
    "2024-03-11",
    "2024-09-30"
]

DISC_BP_DATES = pd.to_datetime(
    DISC_BP_DATES,
    format="%Y-%m-%d",
    errors = "raise"
)
REV_PER_STORE = pd.read_csv("data/per_store/revenue_per_store.csv", index_col = False)




REV_PER_STORE["Sales.Date"] = pd.to_datetime(
    REV_PER_STORE["Sales.Date"],
    format="%Y-%m-%d",
    errors="raise"
)

REV_BP_IDX = [REV_PER_STORE.loc[REV_PER_STORE["Sales.Date"] == REV_BP_DATES[i]].index[0] for i in range(len(REV_BP_DATES))]


def read_data(dir = "data/panels/store/trimmed_store_panels.csv", individual = "Store.Code", verbose = False):
    # Example unbalanced panel data
    df = pd.read_csv(dir, index_col = False)

    # Sort by entity and time
    df = df.sort_values([individual, 'Sales.Date']).reset_index(drop=True)

    df["Sales.Date"] = pd.to_datetime(
        df["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )    
    if verbose:
        print(df.head())
        print(df.shape)
    return df
        
        
# Create lag features
def create_panel_lags(df, group_col, time_col, value_cols, lags=[1,2,3]):
    df = df.sort_values([group_col, time_col])
    for col in value_cols:
        for lag in lags:
            df[f'{col}.lag{lag}'] = df.groupby(group_col)[col].shift(lag)
    return df

# Create rolling features
def create_panel_rolling(df, group_col, value_cols, windows=[3,7]):
    
    if isinstance(windows, int):
        windows = [windows]
    for col in value_cols:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = (
                df.groupby(group_col)[col]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df

        
def prepare_data(dir = "data/combined_store_panel.csv", panel_lags = [1,2,3,4], rolling_windows = [3,7], individual = "Store.Code", scale = False, drop_na = True):
    df = read_data(dir=dir, individual = individual)
    if scale:
        df["Revenue"] = df["Revenue"] / 1000
        df["Discount"] = df["Discount"] / 1000
    if panel_lags is not None:
        df = create_panel_lags(df, individual, 'Sales.Date', ['Revenue'], lags=panel_lags)
        df = create_panel_lags(df, individual, 'Sales.Date', ['Discount'], lags=panel_lags)
    if rolling_windows is not None:
        df = create_panel_rolling(df, individual, 'Sales.Date', ['Revenue'], windows=rolling_windows)
        df = create_panel_rolling(df, individual, 'Sales.Date', ['Discount'], windows=rolling_windows)
    

    feature_cols = [col for col in df.columns if col not in ['Store.Code', 'Sales.Date', 'Revenue', 'Discount']]
    if drop_na:
        df_clean = df.dropna()
    else:
        df_clean = df
    return df_clean, feature_cols


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


with open('data/metadata.json', 'r') as file:
    ENCODINGS = json.load(file) # Parses JSON file into Python dictionary/list

def date_to_int(date_string, fmt="%Y-%m-%d"):
    date = datetime.datetime.strptime(date_string, fmt)
    return (date - ORIGIN).days

def get_panel_with_dummy(original_panel_dir = "data/panels/store/combined_store_panel.csv", individual = "Store.Code"):
    panels_with_dummies = []
    original_panel = pd.read_csv(original_panel_dir, index_col=False)
    original_panel["Sales.Date"] = pd.to_datetime(
        original_panel["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )
    for code in list(ENCODINGS[individual].keys()):
        panel = original_panel.loc[original_panel[individual] == int(code)]
        panel_size = panel.shape[0]
        panel_dates = [panel["Sales.Date"].iloc[i].strftime("%Y-%m-%d") for i in range(panel_size)]
        
        panel_days = np.array([
            datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
            for date_str in panel_dates
        ])
        
        sd_array = np.array([(panel_days == DAYS_OF_WEEK[i]).astype(int) for i in range(len(DAYS_OF_WEEK) - 1)]).transpose()
        sd_df = pd.DataFrame(sd_array, columns = [f"sd{i}" for i in range(1, 7)])
        
        ld_array = np.zeros((panel_size, len(REV_BP_DATES)))
        td_array = np.zeros((panel_size, 1 + len(REV_BP_DATES)))
        
        
        
        
        for i in range(len(REV_BP_DATES)):
            ld_array[:,i] = (np.array(panel_dates, dtype = str) >= REV_BP_DATES[i].strftime("%Y-%m-%d")).astype(int)
            td_array[:,i+1] = np.maximum(0, np.arange(1, panel_size + 1) - (REV_BP_IDX[i]) + date_to_int(panel_dates[0]))
            pass
        
        ld_df = pd.DataFrame(ld_array, columns = [f"ld{i}" for i in range(1, len(REV_BP_DATES) + 1)])
        td_df = pd.DataFrame(td_array, columns = [f"td{i}" for i in range(0, len(REV_BP_DATES) + 1)])
        td_array[:,0] = np.arange(1, panel_size + 1) + date_to_int(panel_dates[0])
        new_panel = pd.concat((panel.reset_index(drop = True), sd_df, ld_df, td_df), axis = 1)
        
        panels_with_dummies.append(new_panel)

        
        
    panels_with_dummies = pd.concat(
        panels_with_dummies, axis=0, ignore_index=True
    )
    
    return panels_with_dummies

def main():
    sgor = pd.read_csv("data/selangor.csv", index_col = False)
    
    sgor_clean = pd.read_csv("data/sgor_clean.csv", index_col = False)
    revenue_per_store = pd.read_csv("data/per_store/revenue_per_store.csv", index_col = False)
    revenue_per_drink = pd.read_csv("data/per_drink/revenue_per_drink.csv", index_col = False)

    discount_per_store = pd.read_csv("data/per_store/discount_per_store.csv", index_col = False)
    discount_per_drink = pd.read_csv("data/per_drink/discount_per_drink.csv", index_col = False)
    combined_store_panel = pd.read_csv("data/panels/store/combined_store_panel.csv", index_col = False)
    combined_drink_panel = pd.read_csv("data/panels/drink/combined_drink_panel.csv", index_col = False)
    
    revenue_per_store["Sales.Date"] = pd.to_datetime(
        revenue_per_store["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )

    revenue_per_drink["Sales.Date"] = pd.to_datetime(
        revenue_per_drink["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )

    discount_per_store["Sales.Date"] = pd.to_datetime(
        discount_per_store["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )

    discount_per_drink["Sales.Date"] = pd.to_datetime(
        discount_per_drink["Sales.Date"],
        format="%Y-%m-%d",
        errors="raise"
    )

    combined_store_panel["Sales.Date"] = pd.to_datetime(
        combined_store_panel["Sales.Date"],
        format = "%Y-%m-%d",
        errors = "raise"
    )

    combined_drink_panel["Sales.Date"] = pd.to_datetime(
        combined_drink_panel["Sales.Date"],
        format = "%Y-%m-%d",
        errors = "raise"
    )

    
    combined_store_panels_lags, feature_cols = prepare_data(
        dir = "data/panels/store/combined_store_panel.csv",
        panel_lags = [1,2,3,4,5,6,7],
        rolling_windows = None,
        scale = False,
        drop_na = False
    )
    print(combined_store_panels_lags.head(5))
    store_to_city = {}
    for city in sgor_clean["City"].unique():
        stores_in_city = sgor_clean.loc[sgor_clean["City"] == city, "Store.Code"].unique()
        for store in stores_in_city:
            store_to_city[store] = city
        print(f"City: {city}, Stores: {stores_in_city}")
        
    store_to_city = dict(sorted(store_to_city.items(), key = lambda item: item[0]))

    store_to_city = {int(k): v for k, v in store_to_city.items()}

    combined_store_panels_lags["City"] = combined_store_panels_lags["Store.Code"].map(store_to_city)


    with open("data/store_to_city.json", "w") as file:
        json.dump(store_to_city, file, indent=4, cls=NumpyEncoder)

    combined_store_panels_lags.to_csv("data/panels/store/combined_store_panels_lags.csv", index = False)
    
    
    
    combined_drink_panel_lags, feature_cols = prepare_data(
        dir = "data/panels/drink/combined_drink_panel.csv",
        panel_lags = [1,2,3,4,5,6,7],
        rolling_windows = None,
        individual = "Product.Name",
        scale = False,
        drop_na = False
    )
    
    combined_drink_panel_lags.to_csv("data/panels/drink/combined_drink_panel_lags.csv", index = False)
    
    
if __name__ == "__main__":
    main()