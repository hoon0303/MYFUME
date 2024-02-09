########################################
# 작성자 : 20164006-박훈
# 프로그램명 : 최근접 이웃 사용자 추천
# similar_users : 사용자 유사도 계산 함수
# recommend_item : 아이템 추천 함수
########################################

import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import correlation
import operator
from sklearn.neighbors import NearestNeighbors

def similar_users(user_id, matrix, k):
    model_knn = NearestNeighbors(metric = 'correlation', algorithm = 'brute')#피어슨 유사도 계산
    model_knn.fit(matrix)#모델 학습

    query_index = matrix.index.get_loc(user_id)# 유사도 대상 쿼리 인덱스
        
    KN = matrix.iloc[query_index].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(KN, n_neighbors = k)#인접한 k개의 sample에 대한 거리 index 반환

    Rec_Users = list()# 유사한 유저 아이디 저장
    User_dis = list()# 유사한 유저 거리 저장
        
    for i in range(1, len(distances.flatten())):# 유사한 사용자 개수만큼 반복
        Rec_Users.append(matrix.index[indices.flatten()[i]])# 유사한 유저 아이디 리스트 저장
        User_dis.append(distances.flatten()[i])# 유사한 유저 거리 리스트 저장

    return Rec_Users# 유사한 유저 리스트 반환

def recommend_item(user_index, similar_user_indices, matrix, items=5):
    
    # 유사한 유저 행렬 필터링
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    
    # 유사 사용자의 정보로 아이템의 평균 예측 점수를 구함
    similar_users = similar_users.mean(axis=0)
    # 행렬에서 평균 점수만 저장
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
    # 행렬에서 사용자 행렬을 저장
    user_df = matrix[matrix.index == user_index]
    # 행 열 변환
    user_df_transposed = user_df.transpose()
    # 열의 이름을 rating으로 설정
    user_df_transposed.columns = ['rating']
    # 사용자가 평가하지 않은 향수 정보 저장
    user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
    # 사용자가 평가하지 않은 향수 인덱스 저장
    Not_rating_list = user_df_transposed.index.tolist()
    
    # 사용자가 평가하지 않은 향수 정보를 필터링
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(Not_rating_list)]
    # 아이템 예측 점수 정렬
    similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)
    # 상위 아이템 추출  
    Top_n_perfume = similar_users_df_ordered.head(items)
    # 추천 아이템 인덱스 리스트 저장
    top_n_anime_indices = Top_n_perfume.index.tolist()
    # 추천 아이템 향수 데이터의 정보와 합침
    result_item = perfumes[perfumes['N_id'].isin(top_n_anime_indices)]
    
    return result_item #아이템 추천 리스트

perfumes = pd.read_csv('./data/Perfume_data.csv')# 향수 데이터
ratings = pd.read_csv('./data/Perfume_review1.csv')# 향수 리뷰 데이터

userId = 70000 # 유사 유저 대상 아이디

rating_merge = pd.merge(ratings, perfumes,on="N_id")# 향수 리뷰에 향수 정보를 합침

# 사용자 향기 평균 테이블을 만듬
User_category = rating_merge.pivot_table("PerfumeScore", index= "N_User",columns="Smell",aggfunc="mean")# 피봇 테이블 생성
User_category_matrix = User_category.fillna(0)# 평가가 없는 경우 0으로 채움

print(User_category_matrix)

similar_user_indices = similar_users(userId, User_category_matrix,10000)# 향기 평가가 유사한 사용자 10000명 선발
similar_user_indices.append(userId)# 리스트에 사용자 아이디 추가

Perfume_ratings = ratings[ratings.N_User.isin(similar_user_indices)]# 향수 평가에서 유사 사용자 10000명 필터링

# 사용자 아이템 평가 테이블을 만듬
rating_matrix = Perfume_ratings.pivot_table(index='N_User', columns='N_id', values='PerfumeScore')# 피봇 테이블 생성
rating_matrix = rating_matrix.fillna(0)# 평가가 없는 경우 0으로 채움

similar_user_indices = similar_users(userId, rating_matrix,100)# 향수 평가 정보와 유사한 사용자 100명 선발

result_item = recommend_item(userId, similar_user_indices, rating_matrix)# 아이템 추천
print("사용자 추천 아이디 : %d"%(userId))
print()
print("###추천 결과###")
print(result_item)# 아이템 추천 결과 출력

