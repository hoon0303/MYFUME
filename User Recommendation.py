import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_similar_users(user_id, rating_matrix, num_neighbors):
    """
    주어진 사용자에 대해 유사한 사용자를 찾는 함수.

    Parameters:
        user_id: 유사도를 계산할 사용자 ID.
        rating_matrix: 사용자-아이템 평점 행렬.
        num_neighbors: 반환할 유사한 사용자의 수.

    Returns:
        list: 유사한 사용자 ID 리스트.
    """
    knn_model = NearestNeighbors(metric='correlation', algorithm='brute')
    knn_model.fit(rating_matrix)

    user_index = rating_matrix.index.get_loc(user_id)  # 대상 사용자 인덱스
    user_vector = rating_matrix.iloc[user_index].values.reshape(1, -1)
    
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=num_neighbors)

    similar_user_ids = [rating_matrix.index[i] for i in indices.flatten()[1:]]
    return similar_user_ids

def recommend_items(user_id, similar_user_ids, rating_matrix, top_n=5):
    """
    유사한 사용자들의 평점을 기반으로 추천 아이템을 반환하는 함수.

    Parameters:
        user_id: 추천을 받을 사용자 ID.
        similar_user_ids: 유사한 사용자 ID 리스트.
        rating_matrix: 사용자-아이템 평점 행렬.
        top_n: 추천할 아이템의 수.

    Returns:
        pd.DataFrame: 추천 아이템의 데이터프레임.
    """
    similar_users_ratings = rating_matrix[rating_matrix.index.isin(similar_user_ids)]
    
    # 유사한 사용자들의 평균 평점 계산
    mean_ratings = similar_users_ratings.mean(axis=0)
    mean_ratings_df = pd.DataFrame(mean_ratings, columns=['mean'])

    # 대상 사용자가 평가하지 않은 아이템만 필터링
    user_ratings = rating_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index.tolist()
    
    # 미평가 아이템 중에서 높은 평균 평점을 가진 아이템 추천
    recommendations = mean_ratings_df.loc[mean_ratings_df.index.isin(unrated_items)]
    recommendations = recommendations.sort_values(by='mean', ascending=False).head(top_n)
    
    recommended_items = perfumes[perfumes['N_id'].isin(recommendations.index)]
    return recommended_items

# 데이터 불러오기
perfumes = pd.read_csv('./data/Perfume_data.csv')  # 향수 데이터
ratings = pd.read_csv('./data/Perfume_review1.csv')  # 향수 리뷰 데이터

user_id = 70000  # 유사 사용자 대상 ID

# 사용자-향기 평점 행렬 생성
rating_merge = pd.merge(ratings, perfumes,on="N_id")
user_scent_matrix = rating_merge.pivot_table("PerfumeScore", index="N_User", columns="Smell", aggfunc="mean").fillna(0)

# 유사 사용자 찾기
similar_user_ids = find_similar_users(user_id, user_scent_matrix, 10000)
similar_user_ids.append(user_id)

# 유사 사용자들의 평점만 필터링
filtered_ratings = ratings[ratings.N_User.isin(similar_user_ids)]

# 사용자-아이템 평점 행렬 생성
user_item_matrix = filtered_ratings.pivot_table(index='N_User', columns='N_id', values='PerfumeScore').fillna(0)

# 아이템 추천
final_similar_users = find_similar_users(user_id, user_item_matrix, 100)
recommended_items = recommend_items(user_id, final_similar_users, user_item_matrix)

# 결과 출력
print(f"사용자 추천 ID: {user_id}")
print("\n### 추천 결과 ###")
print(recommended_items)
