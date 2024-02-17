########################################
# Hoon Park
# Project : User Recommender
# similar_users : Similar User Function
# recommend_item : Item Recommendation Function
########################################

import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import correlation
import operator
from sklearn.neighbors import NearestNeighbors
import timeit

def similar_users(user_id, matrix, k):
    model_knn = NearestNeighbors(metric = 'correlation', algorithm = 'brute')
    model_knn.fit(matrix)

    query_index = matrix.index.get_loc(user_id)# Similar query_index
        
    KN = matrix.iloc[query_index].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(KN, n_neighbors = k)#Top k distances, ,index

    Rec_Users = list()# Similar user indices
    User_dis = list()# Similar distances
        
    for i in range(1, len(distances.flatten())):
        Rec_Users.append(matrix.index[indices.flatten()[i]])
        User_dis.append(distances.flatten()[i])

    return Rec_Users# Similar user list

def recommend_item(user_index, similar_user_indices, matrix, items=5):
    
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    
    similar_users = similar_users.mean(axis=0)
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
    user_df = matrix[matrix.index == user_index]
    user_df_transposed = user_df.transpose()
    user_df_transposed.columns = ['rating']
    user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
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

userId = 9570 # 유사 유저 대상 아이디

rating_merge = pd.merge(ratings, perfumes,on="N_id")# 향수 리뷰에 향수 정보를 합침

# 사용자 향기 평균 테이블을 만듬
User_category = rating_merge.pivot_table("PerfumeScore", index= "N_User",columns="Smell",aggfunc="mean")# 피봇 테이블 생성
User_category_matrix = User_category.fillna(0)# 평가가 없는 경우 0으로 채움

rating_merge = rating_merge[ rating_merge.Smell == 'Aromatic']

similar_user_indices = similar_users(userId, User_category_matrix,1000)# 향기 평가가 유사한 사용자 10000명 선발
similar_user_indices.append(userId)# 리스트에 사용자 아이디 추가

Perfume_ratings = ratings[ratings.N_User.isin(similar_user_indices)]# 향수 평가에서 유사 사용자 10000명 필터링

x = 1000
# 사용자 아이템 평가 테이블을 만듬

start_time = timeit.default_timer() # 시작 시간 체크

rating_matrix = Perfume_ratings.pivot_table(index='N_User', columns='N_id', values='PerfumeScore')# 피봇 테이블 생성
rating_matrix = rating_matrix.fillna(0)# 평가가 없는 경우 0으로 채움

similar_user_indices = similar_users(userId, rating_matrix,100)# 향수 평가 정보와 유사한 사용자 100명 선발

result_item = recommend_item(userId, similar_user_indices, rating_matrix)# 아이템 추천
print("사용자 추천 아이디 : %d"%(userId))
print()
print("###추천 결과###")
print(result_item)# 아이템 추천 결과 출력
terminate_time = timeit.default_timer() # 종료 시간 체크v
print()
print("%d개의 데이터 %f초 걸렸습니다."%(x,terminate_time - start_time)) 

x = 500000
Perfume_ratings = ratings[:x]
# 사용자 아이템 평가 테이블을 만듬

start_time = timeit.default_timer() # 시작 시간 체크

rating_matrix = Perfume_ratings.pivot_table(index='N_User', columns='N_id', values='PerfumeScore')# 피봇 테이블 생성
rating_matrix = rating_matrix.fillna(0)# 평가가 없는 경우 0으로 채움

similar_user_indices = similar_users(userId, rating_matrix,100)# 향수 평가 정보와 유사한 사용자 100명 선발

result_item = recommend_item(userId, similar_user_indices, rating_matrix)# 아이템 추천
print("사용자 추천 아이디 : %d"%(userId))
print()
print("###추천 결과###")
print(result_item)# 아이템 추천 결과 출력
terminate_time = timeit.default_timer() # 종료 시간 체크v
print()
print("%d개의 데이터 %f초 걸렸습니다."%(x,terminate_time - start_time)) 

