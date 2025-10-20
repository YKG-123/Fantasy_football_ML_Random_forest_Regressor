from python.df_to_json import df_to_json
import pandas as pd


def main():
    df = pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Allen, Mr. William Henry",
                "Bonnel, Miss. Elizabeth",
            ],
            "Age": [22, 35, 58],
            "Sex": ["male", "male", "female"],
        }
    )

    df_to_json(df, output_file="dump/test.json")


if __name__ == "__main__":
    main()
