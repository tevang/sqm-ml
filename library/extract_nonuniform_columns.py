
def extract_nonuniform_columns(df, selected_columns):
    return df.columns[df.columns.isin(selected_columns)].to_list()