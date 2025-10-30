import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import logging

from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models


try:
    from prophet import Prophet
    USE_PROPHET = True
except ImportError:
    from statsmodels.tsa.arima.model import ARIMA
    USE_PROPHET = False


os.makedirs("../outputs", exist_ok=True)
os.makedirs("../models", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("insights")


def load_feedback_data():
    """Load available feedback data from expected directories."""
    search_paths = [
        "../data/feedback_with_summaries.csv",
        "../data/simulated_feedback.csv",
        "../data/cleaned_feedback.csv"
    ]

    for path in search_paths:
        if os.path.exists(path):
            logger.info(f"Loaded data from {path}")
            df = pd.read_csv(path)
            return df

    raise FileNotFoundError("Could not locate any feedback data CSV file.")


def prepare_daily_scores(df):
    """Prepare daily average satisfaction scores for forecasting."""
    df = df.copy()

    if "rating" in df.columns:
        df["score"] = pd.to_numeric(df["rating"], errors="coerce")
    elif "sentiment" in df.columns:
        sentiment_map = {"positive": 5, "neutral": 3, "negative": 1}
        df["score"] = df["sentiment"].map(sentiment_map).fillna(3)
    else:
        df["score"] = 3  # neutral default

    if "date" not in df.columns:
        today = pd.Timestamp.today()
        df["date"] = [today - timedelta(days=i) for i in range(len(df))][::-1]

    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")

    daily = (
        df.groupby("date")["score"]
        .mean()
        .reset_index()
        .rename(columns={"date": "ds", "score": "y"})
    )

    return daily, df


def perform_topic_modeling(df, num_topics=5):
    """Run a simple LDA topic modeling on feedback text."""
    text_col = None
    for c in ["feedback_text", "feedback", "text"]:
        if c in df.columns:
            text_col = c
            break

    if not text_col:
        logger.warning("No feedback text column found for topic modeling.")
        return []

    texts = df[text_col].astype(str).tolist()
    tokenized = [t.split() for t in texts]

    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]

    lda_model = models.LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=8, random_state=42
    )

    topics = lda_model.print_topics(num_topics=num_topics, num_words=6)
    lda_model.save("../models/lda_model")


    plt.figure(figsize=(8, len(topics) * 1.3))
    txt = "\n".join([f"Topic {tid}: {t}" for tid, t in topics])
    plt.text(0, 0.5, txt, fontsize=11)
    plt.axis("off")
    plt.title("LDA Topic Summary")
    plt.savefig("../outputs/lda_topics.png", bbox_inches="tight")
    plt.close()

    return topics


def forecast_scores(daily_df, days_ahead=30):
    """Forecast satisfaction trend using Prophet or ARIMA."""
    if USE_PROPHET:
        try:
            model = Prophet()
            model.fit(daily_df)
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)

            fig = model.plot(forecast)
            fig.savefig("../outputs/prophet_forecast.png")
            plt.close(fig)

            return forecast

        except Exception as e:
            logger.error(f"Prophet forecasting failed: {e}")


    logger.info("Using ARIMA as fallback model.")
    ts = daily_df.set_index("ds")["y"].asfreq("D").fillna(method="ffill")

    model = ARIMA(ts, order=(5, 1, 0))
    res = model.fit()
    pred = res.get_forecast(steps=days_ahead)
    ci = pred.conf_int()

    future_dates = [ts.index.max() + timedelta(days=i) for i in range(1, days_ahead + 1)]
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": pred.predicted_mean,
        "yhat_lower": ci.iloc[:, 0],
        "yhat_upper": ci.iloc[:, 1],
    })

    plt.figure(figsize=(10, 5))
    plt.plot(ts.index, ts.values, label="Historical")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="gray", alpha=0.2)
    plt.legend()
    plt.title("ARIMA Forecast (Average Satisfaction)")
    plt.savefig("../outputs/arima_forecast.png", bbox_inches="tight")
    plt.close()

    return forecast


def main():
    df = load_feedback_data()
    daily, full = prepare_daily_scores(df)
    logger.info(f"Prepared daily satisfaction series with {len(daily)} records")


    topics = perform_topic_modeling(full, num_topics=5)
    if topics:
        for tid, desc in topics:
            logger.info(f"Topic {tid}: {desc}")

    
    forecast = forecast_scores(daily, days_ahead=30)

    
    summary_lines = [
        f"Records processed: {len(df)}",
        f"Forecast days: {30}",
        "Top topics found:"
    ]
    for tid, desc in topics:
        summary_lines.append(f"- Topic {tid}: {desc}")

    with open("../outputs/insight_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    logger.info("Insight generation completed. Results saved in '../outputs/'.")


if __name__ == "__main__":
    main()
