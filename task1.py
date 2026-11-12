import os
import requests
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Беремо саме той файл, який ти згенерував у Практичній роботі №5
LAB5_CSV = os.path.join(CURRENT_DIR, "gdp_per_capita_2019.csv")
OUTPUT_CSV = os.path.join(CURRENT_DIR, "analyzed_gdp_2019.csv")


def enrich_lab5_data(df: pd.DataFrame) -> pd.DataFrame:
    # Доповнює DataFrame з 5-ї лаби двома новими стовпцями: Населення та Регіон. 
    print("[INFO] Доповнення набору даних показниками Population та Region через API...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url_pop = "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date=2019&format=json&per_page=300"
        url_meta = "https://api.worldbank.org/v2/country?format=json&per_page=300"

        res_pop = requests.get(url_pop, headers=headers, timeout=30).json()[1]
        res_meta = requests.get(url_meta, headers=headers, timeout=30).json()[1]

        pop_map = {
            item["countryiso3code"]: item["value"]
            for item in res_pop
            if item.get("value") is not None
        }
        region_map = {
            item["id"]: item["region"]["value"]
            for item in res_meta
            if item.get("region", {}).get("value") != "Aggregates"
        }

        # Додаємо два нові стовпці до нашого DataFrame за кодом країни (Country Code)
        df["Region"] = df["Country Code"].map(region_map)
        df["Population"] = df["Country Code"].map(pop_map)

        # Видаляємо агреговані регіони (типу 'Arab World', 'World'), де Region не визначився
        df = df.dropna(subset=["Region", "Population"]).copy()
        df["Population"] = df["Population"].astype("int64")

        print("[УСПІХ] Набір даних успішно доповнено новими стовпцями!")
        return df

    except Exception as err:
        print(f"[ПОМИЛКА API] Не вдалося завантажити додаткові дані: {err}")
        return df


def main():
    # 1. Завантаження твого CSV-файлу з Практичної роботи №5
    try:
        df = pd.read_csv(LAB5_CSV)
        print(f"[УСПІХ] Файл '{LAB5_CSV}' з Практичної 5 успішно завантажено!")
    except FileNotFoundError:
        print(f"[ПОМИЛКА] Файл '{LAB5_CSV}' не знайдено! Скопіюйте його з папки lab_5-6.")
        return

    # Уніфікуємо назву стовпця ВВП (у тебе в 5-й лабі він називався 'GDP per Capita (US$)')
    gdp_col = [col for col in df.columns if "gdp" in col.lower()][0]
    df = df.rename(columns={gdp_col: "GDP_per_capita"})

    # 2. Доповнення даними (Population та Region)
    df = enrich_lab5_data(df)

    # 3. Базовий аналіз одержаних даних
    print("\n" + "=" * 60)
    print("БАЗОВИЙ АНАЛІЗ DATAFRAME")
    print("=" * 60)
    print("\n--- Перші 3 рядки DataFrame (head(3)) ---")
    print(df.head(3))

    print("\n--- Типи даних у стовпцях (dtypes) ---")
    print(df.dtypes)

    print(f"\n--- Розмір DataFrame (shape): {df.shape[0]} рядків, {df.shape[1]} стовпців ---")

    print("\n--- Описова статистика для числових даних (describe()) ---")
    print(df.describe())

    # 4. Попереднє оброблення: додавання нового розрахункового стовпця
    # Загальний номінальний ВВП країни у мільярдах доларів
    df["Total_GDP_Billion_USD"] = round((df["GDP_per_capita"] * df["Population"]) / 1e9, 2)
    print("\n--- DataFrame з розрахунковим стовпцем Total_GDP_Billion_USD ---")
    print(df[["Country Name", "Region", "GDP_per_capita", "Total_GDP_Billion_USD"]].head(5))

    # 5. Фільтрація даних (країни з ВВП на душу населення > 20 000$)
    filtered_df = df[df["GDP_per_capita"] > 20000]
    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТ ФІЛЬТРАЦІЇ (GDP_per_capita > 20 000$): знайдено {len(filtered_df)} країн")
    print("=" * 60)
    print(filtered_df[["Country Name", "Region", "GDP_per_capita"]].head(10))

    # 6. Сортування даних за спаданням ВВП на душу населення
    sorted_df = filtered_df.sort_values(by="GDP_per_capita", ascending=False)
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ СОРТУВАННЯ (ТОП-10 країн за GDP per capita):")
    print("=" * 60)
    print(sorted_df[["Country Name", "Region", "GDP_per_capita", "Total_GDP_Billion_USD"]].head(10))

    # 7. Групування та агрегація даних за регіонами
    grouped_df = df.groupby("Region").agg(
        Countries_Count=("Country Name", "count"),
        Mean_GDP_per_capita=("GDP_per_capita", "mean"),
        Max_GDP_per_capita=("GDP_per_capita", "max"),
        Min_GDP_per_capita=("GDP_per_capita", "min")
    ).round(2).reset_index()

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ ГРУПУВАННЯ ТА АГРЕГАЦІЇ ЗА РЕГІОНАМИ:")
    print("=" * 60)
    print(grouped_df)
    print(f"\nКількість унікальних регіонів: {df['Region'].nunique()}")

    # 8. Збереження результатів у новий CSV-файл
    sorted_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[УСПІХ] Результати аналізу збережено у файл '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()