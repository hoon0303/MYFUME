import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def calculate_similarity(target_id, data_matrix, num_neighbors):
    """
    주어진 아이템에 대해 최근접 이웃을 찾아 유사한 아이템을 반환하는 함수.

    Parameters:
        target_id: 유사도를 계산할 대상의 ID.
        data_matrix: 유사도 계산에 사용할 데이터 매트릭스.
        num_neighbors: 반환할 이웃의 수.

    Returns:
        list: 유사한 아이템의 ID 리스트.
    """
    knn_model = NearestNeighbors(metric='correlation', algorithm='brute')
    knn_model.fit(data_matrix)

    query_index = data_matrix.index.get_loc(target_id)  # 대상 쿼리 인덱스
    query_vector = data_matrix.iloc[query_index].values.reshape(1, -1)
    
    distances, indices = knn_model.kneighbors(query_vector, n_neighbors=num_neighbors)

    similar_ids = [data_matrix.index[i] for i in indices.flatten()[1:]]
    return similar_ids

# 데이터 불러오기
perfumes_df = pd.read_csv('./data/Perfume_data.csv')  # 향수 데이터
ratings_df = pd.read_csv('./data/Perfume_review1.csv')  # 향수 리뷰 데이터

target_perfume_id = 23535  # 추천할 향수의 ID

# 대상 향수 정보 출력
print(f"향수 추천 대상 ID: {target_perfume_id}")
target_perfume_info = perfumes_df[perfumes_df.N_id == target_perfume_id]
print("\n향수 추천 대상 정보")
print(target_perfume_info)
print()

# 평점 데이터와 향수 데이터를 병합
merged_ratings = pd.merge(ratings_df, perfumes_df, on="N_id")
target_smell = perfumes_df[perfumes_df.N_id == target_perfume_id]['Smell'].values[0]

# 평균 평점 매트릭스 생성
mean_ratings_matrix = merged_ratings.pivot_table("PerfumeScore", index="Smell", columns="N_User", aggfunc="mean")
mean_ratings_matrix = mean_ratings_matrix.fillna(0)

# 유사한 향기의 향수 찾기
similar_smells = calculate_similarity(target_smell, mean_ratings_matrix, 3)
similar_smells.append(target_smell)
print("### 해당 향수와 유사한 향기 ###")
print(similar_smells)
print()

# 유사한 향기의 향수들의 평점 데이터 필터링
filtered_ratings = merged_ratings[merged_ratings.Smell.isin(similar_smells)]

# 평점 매트릭스 생성
rating_matrix = filtered_ratings.pivot_table(index='N_id', columns='N_User', values='PerfumeScore')
rating_matrix = rating_matrix.fillna(0)

# 유사한 아이템 추천
recommended_item_ids = calculate_similarity(target_perfume_id, rating_matrix, 9)

# 최종 추천 결과 출력
print("### 최종 추천 ###")
final_recommendations = perfumes_df[perfumes_df.N_id.isin(recommended_item_ids)]
print(final_recommendations)
