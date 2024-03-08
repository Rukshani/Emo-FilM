"""
@author Rukshani Somarathna

This file contains the data fields used in the Emo-FilM project.
"""

# durations of 14 movies
dur_movies = [496, 808, 490, 405, 599, 667, 1008, 722, 805, 1028, 588, 784, 402, 798]

# 14 movies
movies = ['AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'FirstBite', 'LessonLearned',
          'Payload', 'Sintel', 'Spaceman', 'Superhero', 'TearsofSteel', 'Thesecretnumber',
          'ToClaireFromSonny', 'YouAgain']

# 13 discrete_items
sorted_discrete_items = ['Love', 'Regard', 'WarmHeartedness', 'Satisfaction', 'Happiness', 'Pride',
                         'Anxiety', 'Fear', 'Surprise', 'Sad', 'Disgust', 'Anger', 'Guilt']

# 50 sorted all items
sorted_all_items = ['incongruent with your standards_A', 'unpleasant for you_A',
                    'violated laws/norms_A', 'unpleasant for other_A',
                    'important for the goals of other_A', 'uncontrolled_A',
                    'unpredictable_A', 'occurred suddenly_A', 'agent_A', 'no urgency_A',
                    'pressed lips together_E', 'had tears_E', 'eyebrows go up_E', 'smiled_E',
                    'frowned_E', 'urge to stop what was happening_M', 'undo what was happening_M',
                    'ongoing situation to last/repeat_M', 'oppose someone or something_M',
                    'motivated to pay attention_M', 'tackling the situation_M',
                    'wanted to be in command of others_M', 'someone to be there to help_M',
                    'wanted to move_M', 'take care of another person or cause_M', 'felt bad_F',
                    'felt good_F', 'felt calm_F', 'felt strong_F', 'intense emotional state_F',
                    'felt tired_F', 'felt at ease_F', 'muscle tensions_P',
                    'experienced heart rate changes_P', 'feeling of a lump in the throat_P',
                    'stomach discomfort_P', 'felt warm_P', 'Anger', 'Guilt', 'WarmHeartedness',
                    'Disgust', 'Happiness', 'Fear', 'Regard', 'Anxiety', 'Satisfaction', 'Pride',
                    'Surprise', 'Love', 'Sad']

# 37 sorted CoreGRID items
sorted_grid_items = sorted_all_items[0:37]
