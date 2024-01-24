# 2023 Samsung AI Challenge : Image Quality Assessment
카메라 영상 화질 정량 평가 및 자연어 정성 평가를 동시 생성하는 알고리즘 개발

<br/>

## 1. 배경 & 목적
 
- Image Quality Assessment & Captioning (화질 평가/캡셔닝)'을 주제로, 카메라로 촬영된 영상의 화질에 대한 정량 평가 점수를 예측하고 그 평가 결과를 자연어로 상세하게 표현하는 알고리즘을 개발
- 화질 평가는 모두가 동의하는 절대적인 기준이 없고, 영상의 선명도, 노이즈 정도, 색감, 선호도 등 다양한 인지 화질 요소를 종합적으로 고려 해야 하는 Challenging 한 문제
- 다양한 인지 화질 요소에 대한 평가를 단일 점수로 나타낼 순 있으나 많은 의미가 생략, 따라서, 새로운 화질 평가 연구의 한 방향으로 자연어로 상세히 영상의 화질을 설명할 수 있는 기술이 필요
- 이 기술은 향후 스마트폰 카메라에서 개인별, 상황별, 국가 별로 특성화되어 사용자에게 최고의 화질을 제공할 수 있는 AI 영상 처리 기술 개발에 활용될 예정

<br/>

## 2. 주최/주관 & 성과

- 주최/주관: 삼성전자 SAIT /DACON
- 참가인원 : 395명
- 성과 : 10등 (본선 진출)
 
<br/>

![스크린샷 2024-01-24 152012](https://github.com/yugwangyeol/2023-Samsung-AI-Challenge/assets/72298825/2935e040-0a26-453a-8105-0f2f2ed789fc)

<br/>

## 3. 프로젝트 기간

- 대회 기간 : 2023년 8월 21일 ~ 10월 2일
- 코드 및 PPT 제출 : 2023년 10월 2일 ~ 10월 6일
- 최종 발표 : 2023년 10월 25일

<br/>

## 4. 프로젝트 소개

&nbsp;&nbsp;&nbsp;&nbsp; 2023년 Samsung AI Challenge : Image captiong 대회에서는 메라로 촬영된 영상의 화질에 대한 정량 평가 점수를 예측하고 그 평가 결과를 자연어로 상세하게 표현하는 알고리즘을 개발을 주제로 대회를 진행하였다. 

![스크린샷 2024-01-24 151115](https://github.com/yugwangyeol/2023-Samsung-AI-Challenge/assets/72298825/85111fea-4bbc-4334-b673-c1f0bd7bb89f)

&nbsp;&nbsp;&nbsp;&nbsp; Baseline으로 주어진 code에서는 해당 Task를 한번에 처리하였으나, 우리 팀은 2개의 Task로 나누어 진행하였다. IQA Task는 PLCC와 SRCC를 사용하여 평가하였고, Captioning 같은 경우에는 총 4개의 평가 지표를 조합하여 평가하였다. 특히 대회 특성상 2개의 Task를 동시에 평가하므로 리더보드에서는 2개의 점수를 합하여 평가를 진행하였다. 추가적으로 동점일 경우 captioning 점수가 높은 팀이 상위로 인정되었다.

&nbsp;&nbsp;&nbsp;&nbsp; 주어진 데이터는 AVA-caption 데이터와 KoniQ++ 데이터가 합쳐진 데이터로 관련된 연구를 많이 조사하였다. IQA Task의 경우 MetaIQA, MANIQA와 같은 IQA 특화 모델을 사용하였으나, 성능이 좋지 않아 Timm에 있는 최신 모델을 사용하여 학습시켜 IQA score를 도출하였다. Captioning Task 경우 BLIP, OFA와 같은 대형 모델을 사용하였으나, 자원의 제한으로 small 모델들만 사용하여 진행하였다. 이를 통해 10등으로 본선에 진출하였다.


<br/>

## 5. Process

### ch.1 Data Preprocessing

- 중복 데이터 제거
- 같은 이미지 다른 captioning 제거

---

### ch.2 Modeling

- IQA model
  - MetaIQA
  - MENIQA
  - Timm (VIT, SWIN-T)

<br/>

## 6. 프로젝트 팀원 및 담당 역할

**[팀원]**

- 학부생 3명

**[담당 역할]**

- 데이터 전처리 및 EDA
- IQA model
- Caption base model

<br/>

## 6. 발표 자료&참고 자료

[2023-Samsung-Ai-Challenge DACON](https://dacon.io/competitions/official/236134/overview/description)  
