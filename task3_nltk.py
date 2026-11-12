import string
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg, stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer


def download_nltk_resources() -> None:
    # Автоматично завантажує необхідні корпуси та словники NLTK.
    resources = ["gutenberg", "stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            print(f"[INFO] Завантаження ресурсу NLTK: {res}...")
            nltk.download(res, quiet=True)


def plot_top_words_bar(top_words: list[tuple[str, int]], title: str, color: str) -> None:
    # Будує стовпчасту діаграму для ТОП-10 слів.
    words = [item[0] for item in top_words]
    counts = [item[1] for item in top_words]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(words, counts, color=color, edgecolor="black", width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Слова (токени)", fontsize=11)
    plt.ylabel("Кількість вживань у тексті", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    download_nltk_resources()

    # 1. Завантаження тексту за Варіантом 1 ('austen-emma.txt')
    text_name = "austen-emma.txt"
    raw_words = gutenberg.words(text_name)

    print("=" * 65)
    print(f"  АНАЛІЗ ТЕКСТУ: {text_name} (ВАРІАНТ 1)")
    print("=" * 65)

    # 2. Загальна кількість слів та кількість унікальних слів
    total_words = len(raw_words)
    unique_words = len(set(word.lower() for word in raw_words))
    lexical_diversity = (unique_words / total_words) * 100

    print(f"\nЗагальна кількість токенів (слів та знаків) у тексті : {total_words}")
    print(f"Кількість унікальних слів (без урахування регістру) : {unique_words}")
    print(f"Коефіцієнт лексичного різноманіття                  : {lexical_diversity:.2f}%")
    print(f"Висновок: загальна кількість слів більша за кількість унікальних у {total_words / unique_words:.1f} раза.")

    # 3. ТОП-10 найбільш уживаних слів у початковому тексті + діаграма
    fdist_raw = FreqDist(raw_words)
    top_10_raw = fdist_raw.most_common(10)

    print("\n" + "-" * 65)
    print("ТОП-10 НАЙБІЛЬШ УЖИВАНИХ ТОКЕНІВ (ДО ОЧИЩЕННЯ):")
    print("-" * 65)
    for word, count in top_10_raw:
        print(f"  '{word:<10}' : {count} разів")

    plot_top_words_bar(
        top_10_raw,
        f"ТОП-10 слів у тексті '{text_name}' (до видалення стоп-слів)",
        color="#FF7043"
    )

    # 4. Видалення стоп-слів та пунктуації
    stop_words = set(stopwords.words("english"))
    table = str.maketrans("", "", string.punctuation)

    cleaned_words = []
    for word in raw_words:
        # Прибираємо пунктуацію та переводимо в нижній регістр
        clean_w = word.translate(table).lower()
        # Залишаємо лише літерні слова, які не входять до стоп-слів
        if clean_w.isalpha() and clean_w not in stop_words:
            cleaned_words.append(clean_w)

    fdist_clean = FreqDist(cleaned_words)
    top_10_clean = fdist_clean.most_common(10)

    print("\n" + "-" * 65)
    print("ТОП-10 НАЙБІЛЬШ УЖИВАНИХ СЛІВ (ПІСЛЯ ВИДАЛЕННЯ СТОП-СЛІВ І ПУНКТУАЦІЇ):")
    print("-" * 65)
    for word, count in top_10_clean:
        print(f"  '{word:<10}' : {count} разів")

    plot_top_words_bar(
        top_10_clean,
        f"ТОП-10 змістовних слів у '{text_name}' (після очищення)",
        color="#42A5F5"
    )

    # 5. Лематизація слів та ТОП-10 після лематизації
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleaned_words]

    fdist_lemma = FreqDist(lemmatized_words)
    top_10_lemma = fdist_lemma.most_common(10)

    print("\n" + "-" * 65)
    print("ТОП-10 НАЙБІЛЬШ УЖИВАНИХ СЛІВ ПІСЛЯ ЛЕМАТИЗАЦІЇ:")
    print("-" * 65)
    for word, count in top_10_lemma:
        print(f"  '{word:<10}' : {count} разів")
    print("=" * 65)


if __name__ == "__main__":
    main()