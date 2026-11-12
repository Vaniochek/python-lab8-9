import os
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BIKE_CSV = os.path.join(CURRENT_DIR, "comptage_velo_2018.csv")


def main():
    # 1. Завантаження CSV-файлу
    try:
        # Файли Монреаля зазвичай мають роздільник ',' або ';', тому використаємо sep=None (автовизначення)
        df = pd.read_csv(BIKE_CSV, sep=None, engine="python", encoding="utf-8")
        print("[УСПІХ] Дані велодоріжок за 2018 рік успішно завантажено!")
    except FileNotFoundError:
        print(f"[ПОМИЛКА] Файл '{BIKE_CSV}' не знайдено!")
        print("Завантажте файл за 2018 рік із порталу donnees.montreal.ca у папку зі скриптом.")
        return
    except Exception as err:
        print(f"[ПОМИЛКА ЧИТАННЯ] {err}")
        return

    # 2. Первинний аналіз DataFrame
    print("\n" + "=" * 60)
    print("ПЕРВИННИЙ АНАЛІЗ DATAFRAME (2018 РІК)")
    print("=" * 60)
    print("\n--- head() ---")
    print(df.head())
    print("\n--- info() ---")
    print(df.info())
    print("\n--- describe() ---")
    print(df.describe())

    # Знаходимо стовпець із датою (зазвичай 'Date' або перший стовпець)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    # Виділяємо лише числові стовпці (лічильники велодоріжок)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Виключаємо службові стовпці часу, якщо вони є
    track_cols = [c for c in numeric_cols if "heure" not in c.lower() and "time" not in c.lower()]

    # Заповнюємо можливі пропуски (NaN) нулями для коректного підрахунку суми
    df[track_cols] = df[track_cols].fillna(0)

    # 3. Загальна кількість велосипедистів за 2018 рік для всіх велодоріжок
    total_cyclists = int(df[track_cols].sum().sum())
    print("\n" + "=" * 60)
    print(f"ЗАГАЛЬНА КІЛЬКІСТЬ ВЕЛОСИПЕДИСТІВ ЗА 2018 РІК (УСІ ДОРІЖКИ): {total_cyclists:,}".replace(",", " "))
    print("=" * 60)

    # 4. Додаємо стовпець місяця та групуємо дані за місяцями
    df["Month"] = df[date_col].dt.month
    monthly_df = df.groupby("Month")[track_cols].sum()

    # Назви місяців для зручного виводу
    month_names = {
        1: "Січень", 2: "Лютий", 3: "Березень", 4: "Квітень",
        5: "Травень", 6: "Червень", 7: "Липень", 8: "Серпень",
        9: "Вересень", 10: "Жовтень", 11: "Листопад", 12: "Грудень"
    }

    # 5. Найбільш популярний місяць для трьох обраних велодоріжок
    selected_tracks = track_cols[:3]  # Беремо перші 3 велодоріжки з набору даних
    print("\n--- Найбільш популярний місяць для 3 обраних велодоріжок ---")
    for track in selected_tracks:
        best_month_num = monthly_df[track].idxmax()
        max_val = int(monthly_df[track].max())
        print(f" * Доріжка '{track}': {month_names.get(best_month_num, best_month_num)} ({max_val:,} велосипедистів)".replace(",", " "))

    # 6. Динаміка використання однієї велодоріжки за місяцями та побудова графіка
    target_track = selected_tracks[0]
    print(f"\n--- Динаміка використання велодоріжки '{target_track}' за місяцями 2018 року ---")
    for m_num, val in monthly_df[target_track].items():
        print(f"Місяць {m_num:02d} ({month_names.get(m_num, '')}): {int(val)}")

    # Побудова графіка за допомогою Matplotlib
    plt.figure(figsize=(10, 5))
    months_x = monthly_df.index.tolist()
    values_y = monthly_df[target_track].values

    plt.plot(months_x, values_y, marker="o", linewidth=2.5, color="#007ACC", label=target_track)
    plt.title(f"Динаміка кількості велосипедистів на доріжці '{target_track}' (2018 рік)", fontsize=13, fontweight="bold")
    plt.xlabel("Місяць", fontsize=11)
    plt.ylabel("Кількість велосипедистів", fontsize=11)
    plt.xticks(months_x, [month_names.get(m, str(m)) for m in months_x], rotation=30)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()