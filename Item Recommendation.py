########################################
# 작성자 : 20164006-박훈
# 프로그램명 : 최근접 이웃 아이템 추천
# similar : 유사도 계산 함수
########################################

import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import correlation
import operator
from sklearn.neighbors import NearestNeighbors

def similar(similar_id, matrix, k):
    model_knn = NearestNeighbors(metric = 'correlation', algorithm = 'brute')#피어슨 유사도 계산
    model_knn.fit(matrix)

    query_index = matrix.index.get_loc(similar_id)# 유사도 대상 쿼리 인덱스
        
    KN = matrix.iloc[query_index].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(KN, n_neighbors = k)#인접한 k개의 sample에 대한 거리 index 반환

    Rec_similar = list()# 유사 아이디 저장
    similar_dis = list()# 유사 거리 저장
        
    for i in range(1, len(distances.flatten())):# 유사 리스트 개수만큼 반복
        Rec_similar.append(matrix.index[indices.flatten()[i]])# 유사 아이디 리스트 저장
        similar_dis.append(distances.flatten()[i])# 유사 거리 리스트 저장

    return Rec_similar# 유사한 리스트 반환

perfumes = pd.read_csv('./data/Perfume_data.csv')# 향수 데이터
ratings = pd.read_csv('./data/Perfume_review1.csv')# 향수 리뷰 데이터

perfumeId = 23535# 향수 아이디

print("향수 추천 아이디 : %d"%(perfumeId))
temp = perfumes[ perfumes.N_id == perfumeId]
print()
print("향수 추천 대상 정보")
print(temp)
print()

rating_merge = pd.merge(ratings, perfumes,on="N_id")
perfumeSemll = perfumes[ perfumes.N_id == perfumeId]['Smell'].values[0]

mean_ratings=rating_merge.pivot_table("PerfumeScore", index= "Smell",columns="N_User",aggfunc="mean")
mean_ratings = mean_ratings.fillna(0)

similar_smell_indices = similar(perfumeSemll, mean_ratings,3)
similar_smell_indices.append(perfumeSemll)
print("###해당 향수와 유사한 향기###")
print(similar_smell_indices)
print()

Perfume_ratings = rating_merge[rating_merge.Smell.isin(similar_smell_indices)]

rating_matrix = Perfume_ratings.pivot_table(index='N_id', columns='N_User', values='PerfumeScore')
rating_matrix = rating_matrix.fillna(0)

similar_Item_indices = similar_user_indices = similar(perfumeId, rating_matrix,9)

print("###최종 추천###")
similar_Item_indices = perfumes[perfumes.N_id.isin(similar_Item_indices)]
print(similar_Item_indices)
