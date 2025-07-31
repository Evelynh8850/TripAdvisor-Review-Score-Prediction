TripAdvisor 評論分數預測：從文字前處理到模型優化

研究目的
透過 TripAdvisor 飯店評論文字資料，預測使用者給予的 1 至 5 分評分，觀察文字內容與評分高低的關聯性。
本研究著重於文字前處理與 TF-IDF 向量化參數的調整，搭配邏輯回歸進行建模，探索如何優化模型分類效能。

運用工具
使用 Python 進行文字資料清洗、詞性轉換與 lemmatization，並應用 TF-IDF 向量化處理文本特徵。
運用 Scikit-learn 建立邏輯回歸多分類模型，實驗不同向量化與建模參數設定。
採用混淆矩陣與 classification report 分析各模型在不同評分分類上的預測表現。

專題結論
調整 TF-IDF 向量化參數（如 max_df、max_features）有助於提升模型在 1、2、3 分等中低分區間的預測能力。
加入 class_weight='balanced' 設定，可改善中低分類別的 recall 與 f1-score，但也會使其他高分類別的預測表現下降。
多數模型在 4、5 分類別預測表現最佳，1–3 分則因樣本不均與語意模糊而準確度偏低。
整體而言，TF-IDF 向量化與建模參數調整皆可帶來指標上的微幅成長，但準確率提升有限。
未來將嘗試導入隨機森林等更進階的分類模型，以改善整體準確度與中低分類別的預測表現。

https://drive.google.com/file/d/1h8kaNuIIQVlqo-rgMOuOrlPkJBOHtHFV/view?usp=sharing
