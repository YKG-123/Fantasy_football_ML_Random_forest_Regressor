import pandas as pd
import json
from datetime import datetime


def df_to_json(
    df: pd.DataFrame,
    *args,
    output_file: str | None = None,
):
    metadata: dict[str, str] = {"last_update": datetime.now().strftime("%Y-%m-%d")}

    data: dict = {"metadata": metadata, "data": df.to_dict(orient="records")}

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(data, indent=4, fp=f)
    else:
        return json.dumps(data, indent=4)

