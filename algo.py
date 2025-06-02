from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.cluster import Kmeans
from scipy.spatial import distance

model = SentenceTransformer('all-MiniLM-L6-v2')  # BERT-based

"""
This algo works based on what one is looking for in a project 
There be certain scale set for a requirements for the project and use the same scale to rate the features of every person
                                          
Helps teams make decisions
        |

 1)Rank projects based on priority 
 2)Based on a percentage every projects gets the top percent who are closely aligned with the project
 3)Remaining individuals a new hierachy system will be created where for every project we will pick the ones who have 
 decent resemmbelance to project needs but more importance given to their diverse backgrounds   




"""


def word_to_embedding(features):
  features = ["AI research project", "Leadership seminar", "UX design sprint"]
  embeddings = model.encode(features)
  pca = PCA(n_components = 3)

  reduced_embeddings = pca.fit_transform(embeddings)
  return reduced_embeddings


def array_size(value):
    count = 0
    for i in value:  
        count+=count
        continue
    return count

def rank_projects(projects):
    """
    This function is responsible for ranking projects based on student preference
    """
    project_ranking = {}

    for i in range(3):
        project = input("Based on given projects, rank the given projects")
        




def team_selection():
    """
    This function is responsible for doing the main task of selecting students for projects
    """
    students = [
        ["coding", "analytical", "leadership"],
        ["finance", "markets", "critical thinking"],
        ["finanical modelling", "helper", "resourceful"],
        ["history","builder","curious"],
        ["computers","hardworking","mamba mentality"],
        
    ]
    projects = ["AI research project", "Financial Strategy Project", "UX design sprint"]

    student_names = ["Student 1", "Student 2", "Student 3","Student 4", "Student 5", "Student 6"]

    # Compute embeddings for students and projects (teams)
    student_embeddings = [word_to_embedding(s) for s in students]
    project_embeddings = word_to_embedding(projects)

    # Example features for demonstration (replace with actual embeddings if needed)
    team_features = np.array([
        [2, 4, 1],
        [3, 2, 1],
        [5, 4, 3]
    ])
    student_features = np.array([
        [5, 1, 2],  # A-Strong coder
        [2, 5, 1],  # B-Strong Leader
        [1, 2, 5],  # C-Strong Designer
        [4, 3, 2],
        [3, 2, 1],
        [1, 3, 5]
    ])

    # Calculate distances between each team and each student    
    team_difference = euclidean_distances(team_features, student_features)

    # Use Hungarian algorithm to maximize total distance (diversity)
    row_ind, col_ind = linear_sum_assignment(-team_difference)  # maximize

    assignments = {team: student for team, student in zip(row_ind, col_ind)}
    print("Most diverse team assignments (by index):", assignments)
    print("Total diversity score (sum of distances):", team_difference[row_ind, col_ind].sum())

    print("\nAssignment details:")
    for team_idx, student_idx in assignments.items():
        print(f"{projects[team_idx]} <-- {student_names[student_idx]} ({students[student_idx]})")
    return assignments

    


    #1- Rank projects from most to least importance(most to least technical)

    #2 - Rank projects with students using k-means
        #1)Each project has to have k-means to find project-student fit
        #2)Pick the top 3 people per project based on that based on that

    ranking = K
    #3)Remaining 50% get re-ranked with more importance given to diverse backgrounds     

def euclidean_distance(point1,point2):
    return distance.euclidean(point1,point2)




def direct_project_student_fit(k, data, centroids, number_of_projects):
    """
    This function allocates the biggest fit to every ranked project and re-ranks remaining giving more importance to diversity
    available_students = {student-1: [2,3,2], student-2....}


    """
    available_projects = {}
    centroids = {}

    for i in range(number_of_projects):
        for point in data.values():
            distance = [euclidean_distance(point, c) for c in centroids]

            cluster_index = np.argmin(distance)
            
            available_projects[cluster_index] = point

            del data[point.key()]

def indirect_project_student_fit(clusters,centroids):
    """
    This function is mainly responsible for arranging students taking into account both diverse skill sets and needs of the project

    """

    
team_selection()


student_1 = ["coding","analytical","leadership"]
student_2 = ["finance","markets","critical thinking"]
student_3 = ["finanical modelling", "helper","resourceful"]



student_features = np.array([
    [5, 1, 2],  # A-Strong coder
    [2, 5, 1],  # B-Leader
    [1, 2, 5],  # C-Designer

])
distances = euclidean_distances(student_features,student_features)
#print(distances)

a, b = np.unravel_index(np.argmax(distances), distances.shape)
#print(a,b)
