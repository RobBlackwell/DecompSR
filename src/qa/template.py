import random
import numpy as np

def get_sentence(object1, relation, object2):
    # down, up, left, right, left-down, right-up, left-up, right-down
    # 0   , 1,  2,    3,      4,        5,        6,          7
    if relation == 0:
        if np.random.randint(0, 2) == 1:
            return object1_below_object2(object1, object2)
        else:
            return object1_over_object2(object2, object1)
    elif relation == 1:
        if np.random.randint(0, 2) == 1:
            return object1_below_object2(object2, object1)
        else:
            return object1_over_object2(object1, object2)
    elif relation == 2:
        if np.random.randint(0, 2) == 1:
            return object1_left_object2(object1, object2)
        else:
            return object1_right_object2(object2, object1)
    elif relation == 3:
        if np.random.randint(0, 2) == 1:
            return object1_left_object2(object2, object1)
        else:
            return object1_right_object2(object1, object2)
    elif relation == 4:
        if np.random.randint(0, 2) == 1:
            return object1_lowerleft_object2(object1, object2)
        else:
            return object1_upright_object2(object2, object1)
    elif relation == 5:
        if np.random.randint(0, 2) == 1:
            return object1_lowerleft_object2(object2, object1)
        else:
            return object1_upright_object2(object1, object2)
    elif relation == 6:
        if np.random.randint(0, 2) == 1:
            return object1_upleft_object2(object1, object2)
        else:
            return object1_lowerright_object2(object2, object1)
    elif relation == 7:
        if np.random.randint(0, 2) == 1:
            return object1_upleft_object2(object2, object1)
        else:
            return object1_lowerright_object2(object1, object2)


def object1_left_object2(object_1, object_2):
    template_candidates = [
        "AAA is to the left of BBB.",
        "AAA is at BBB’s 9 o'clock.",
        "AAA is positioned left to BBB.",
        "AAA is on the left side to BBB.",
        "AAA and BBB are parallel, and AAA on the left of BBB.",
        "AAA is to the left of BBB horizontally.",
        "The object labeled AAA is positioned to the left of the object labeled BBB.",
        "BBB is over there and AAA is on the left.",
        "AAA is placed in the left direction of BBB.",
        "AAA is on the left and BBB is on the right.",
        "AAA is sitting at the 9:00 position of BBB.",
        "AAA is sitting in the left direction of BBB.",
        "BBB is over there and AAA is on the left of it.",
        "AAA is at the 9 o'clock position relative to BBB.",
        "AAA and BBB are parallel, and AAA is to the left of BBB.",
        "AAA and BBB are horizontal and AAA is to the left of BBB.",
        "AAA and BBB are in a horizontal line with AAA on the left.",
        "AAA is to the left of BBB with a small gap between them.",
        "AAA is on the same horizontal plane directly left to BBB.",
        "AAA is to the left of BBB and is on the same horizontal plane.",
        "BBB and AAA are side by side with BBB to the right and AAA to the left.",
        "AAA and BBB are both there with the object AAA is to the left of object BBB.",
        "AAA and BBB are next to each other with BBB on the right and AAA on the left.",
        "AAA presents left to BBB.",
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('AAA', object_1).replace('BBB', object_2)

def object1_right_object2(object_1, object_2):
    template_candidates = [
        "BBB is to the right of AAA.",
        "BBB is at AAA's 3 o'clock.",
        "BBB is positioned right to AAA.",
        "BBB is on the right side to AAA.",
        "AAA and BBB are parallel, and BBB on the right of AAA.", # AAA and BBB are parallel, and AAA on the right of BBB.  changed by Navdeep
        "BBB is to the right of AAA horizontally.",
        "The object labeled BBB is positioned to the right of the object labeled AAA.",
        "AAA is over there and BBB is on the right.",
        "BBB is placed in the right direction of AAA.",
        "BBB is on the right and AAA is on the left.",
        "BBB is sitting at the 3:00 position to AAA.",
        "BBB is sitting in the right direction of AAA.",
        "AAA is over there and BBB is on the right of it.",
        "BBB is at the 3 o'clock position relative to AAA.",
        "AAA and BBB are parallel, and BBB is to the right of AAA.",  #AAA and BBB are parallel, and AAA is to the right of BBB. changed by Navdeep
        "AAA and BBB are horizontal and BBB is to the right of AAA.", #AAA and BBB are horizontal and AAA is to the right of BBB. changed by Navdeep
        "AAA and BBB are in a horizontal line with BBB on the right.",
        "BBB is to the right of AAA with a small gap between them.",
        "BBB is on the same horizontal plane directly right to AAA.",
        "BBB is to the right of AAA and is on the same horizontal plane.",
        "AAA and BBB are side by side with AAA to the left and BBB to the right.",
        "AAA and BBB are both there with the object BBB is to the right of object AAA.", # AAA and BBB are both there with the object AAA is to the right of object BBB. changed by Navdeep
        "AAA and BBB are next to each other with AAA on the left and BBB on the right.",
        "BBB presents right to AAA.",
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('BBB', object_1).replace('AAA', object_2)

def object1_over_object2(object_1, object_2):
    template_candidates = [
        "AAA is over BBB.",
        "AAA is above BBB.",
        "AAA is directly above BBB.",
        "AAA is on top of BBB.",
        "AAA is at BBB's 12 o'clock.",
        "AAA is positioned above BBB.",
        "AAA is on the top side to BBB.",
        "AAA and BBB are parallel, and AAA is over BBB.",
        "AAA is to the top of BBB vertically.",
        "AAA is over there with BBB below.",
        "The object AAA is positioned directly above the object BBB.",
        "BBB is over there and AAA is directly above it.",
        "AAA is placed on the top of BBB.",
        "AAA is on the top and BBB is at the bottom.",
        "AAA is sitting at the 12:00 position to BBB.",
        "AAA is sitting at the top position to BBB.",
        "BBB is over there and AAA is on the top of it.",
        "AAA is at the 12 o'clock position relative to BBB.",
        "AAA and BBB are parallel, and AAA is on top of BBB.",
        "AAA and BBB are vertical and AAA is above BBB.",
        "AAA and BBB are in a vertical line with AAA on top.",
        "AAA is above BBB with a small gap between them.",
        "AAA is on the same vertical plane directly above BBB.",
        "AAA is on the top of BBB and is on the same vertical plane.",
        "AAA and BBB are side by side with AAA on the top and BBB at the bottom.",
        "AAA and BBB are both there with the object AAA above the object BBB.",
        "AAA and BBB are next to each other with AAA on the top and BBB at the bottom.",
        "AAA presents over BBB.",
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('AAA', object_1).replace('BBB', object_2)

def object1_below_object2(object_1, object_2):
    template_candidates = [
        "BBB is under AAA.",
        "BBB is below AAA.",
        "BBB is directly below AAA.",
        "BBB is at the bottom of AAA.",
        "BBB is at AAA's 6 o'clock.",
        "BBB is positioned below AAA.",
        "BBB is at the lower side of AAA.",
        "BBB and AAA are parallel, and BBB is under AAA.",
        "BBB is at the bottom of AAA vertically.",
        "BBB is over there with AAA above.",
        "The object BBB is positioned directly below the object AAA.",
        "AAA is over there and BBB is directly below it.",
        "BBB is placed at the bottom of AAA.", # AAA is placed at the bottom of BBB. - changed by Navdeep
        "BBB is at the bottom and AAA is on the top.",
        "BBB is sitting at the 6:00 position to AAA.",
        "BBB is sitting at the lower position to AAA.",
        "AAA is over there and BBB is at the bottom of it.",
        "BBB is at the 6 o'clock position relative to AAA.",
        "AAA and BBB are parallel, and BBB is below AAA.",
        "BBB and AAA are vertical and BBB is below AAA.",
        "AAA and BBB are in a vertical line with BBB below AAA.",
        "BBB is below AAA with a small gap between them.",
        "BBB is on the same vertical plane directly below AAA.",
        "BBB is at the bottom of AAA and is on the same vertical plane.", # fixed already
        "AAA and BBB are side by side with BBB at the bottom and AAA on the top.",
        "AAA and BBB are both there with the object BBB below the object AAA.",
        "AAA and BBB are next to each other with BBB at the bottom AAA on the top.",
        "BBB presents below AAA." # fixed already
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('BBB', object_1).replace('AAA', object_2)

def object1_lowerleft_object2(object_1, object_2):
    template_candidates = [
        "AAA is on the lower left of BBB.",
        "AAA is to the bottom left of BBB.",
        "The object AAA is lower and slightly to the left of the object BBB.",
        "AAA is on the left side of and below BBB.",
        "AAA is positioned in the lower left corner of BBB.",
        "AAA is lower left to BBB.",
        "AAA is to the bottom-left of BBB.",
        "AAA is below BBB at 7 o'clock.",
        "AAA is positioned down and to the left of BBB.",
        "The object AAA is positioned below and to the left of the object BBB.",
        "AAA is diagonally left and below BBB.",
        "AAA is placed at the lower left of BBB.",
        "AAA is sitting at the lower left position to BBB.",
        "BBB is there and AAA is at the 8 position of a clock face.", #"BBB is there and AAA is at the 10 position of a clock face.", changed by Navdeep
        "AAA is to the left of BBB and below BBB at approximately a 45 degree angle.",
        "AAA is south west of BBB.",
        "AAA is below and to the left of BBB.",
        "The objects AAA and BBB are over there. The object AAA is lower and slightly to the left of the object BBB.",
        "AAA is directly south west of BBB.",
        "AAA is positioned below BBB and to the left.", # BBB is positioned below AAA and to the left. changed by Navdeep
        "AAA is at a 45 degree angle to BBB, in the lower lefthand corner.",
        "AAA is diagonally below BBB to the left at a 45 degree angle.",
        "Object AAA is below object BBB and to the left of it, too.",
        "AAA is diagonally to the bottom left of BBB.",
        "AAA presents lower left to BBB.",
        "If BBB is the center of a clock face, AAA is located between 7 and 8.",
        "AAA is below BBB and to the left of BBB.",
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('AAA', object_1).replace('BBB', object_2)

def object1_upright_object2(object_1, object_2):
    template_candidates = [
        "BBB is on the upper right of AAA.",
        "BBB is to the top right of AAA.",
        "The object BBB is upper and slightly to the right of the object AAA.",
        "BBB is on the right side and top of AAA.",
        "BBB is positioned in the front right corner of AAA.",
        "BBB is upper right to AAA.",
        "BBB is to the top-right of AAA.",
        "BBB is above AAA at 2 o'clock.",
        "BBB is positioned up and to the right of AAA.",
        "The object BBB is positioned above and to the right of the object AAA.",
        "BBB is diagonally right and above AAA.",
        "BBB is placed at the upper right of AAA.",
        "BBB is sitting at the upper right position to AAA.",
        "AAA is there and BBB is at the 2 position of a clock face.",
        "BBB is to the right and above AAA at an angle of about 45 degrees.",
        "BBB is north east of AAA.",
        "BBB is above and to the right of AAA.",
        "The objects BBB and AAA are over there. The object BBB is above and slightly to the right of the object AAA.",
        "BBB is directly north east of AAA.",
        "BBB is positioned above AAA and to the right.",
        "BBB is at a 45 degree angle to AAA, in the upper righthand corner.",
        "BBB is diagonally above AAA to the right at a 45 degree.",
        "Object BBB is above object AAA and to the right of it, too.", # fixed already
        "BBB is diagonally to the upper right of AAA.",                # fixed already
        "BBB presents upper right to AAA.",
        "If AAA is the center of a clock face, BBB is located between 2 and 3.",
        "BBB is above AAA and to the right of AAA.",
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('BBB', object_1).replace('AAA', object_2)

def object1_lowerright_object2(object_1, object_2):
    template_candidates = [
        "AAA is on the lower right of BBB.",
        "AAA is to the bottom right of BBB.",
        "The object AAA is lower and slightly to the right of the object BBB.",
        "AAA is on the right side and below BBB.",
        "BBB is slightly off center to the top left and AAA is slightly off center to the bottom right.",
        "AAA is positioned in the lower right corner of BBB.",
        "AAA is lower right of BBB.",
        "AAA is to the bottom-right of BBB.",
        "AAA is below BBB at 4 o'clock.",
        "AAA is positioned below and to the right of BBB.",
        "The object AAA is positioned below and to the right of the object BBB.",
        "AAA is diagonally right and below BBB.",
        "AAA is placed at the lower right of BBB.",
        "AAA is sitting at the lower right position to BBB.",
        "BBB is there and AAA is at the 5 position of a clock face.",
        "AAA is to the right and below BBB at an angle of about 45 degrees.", #"AAA is to the right and above BBB at an angle of about 45 degrees."-changed by Navdeep
        "AAA is south east of BBB.",
        "AAA is below and to the right of BBB.",
        "The object AAA and BBB are there. The object AAA is below and slightly to the right of the object BBB.",
        "AAA is directly south east of BBB.",
        "AAA is positioned below BBB and to the right.",
        "AAA is at a 45 degree angle to BBB, in the lower righthand corner.",
        "AAA is diagonally below BBB to the right at a 45 degree angle.",
        "Object AAA is below object BBB and to the right of it, too.",
        "AAA is diagonally to the bottom right of BBB.",
        "AAA presents lower right to BBB.",
        "If AAA is the center of a clock face, BBB is located between 10 and 11.",
        "AAA is below BBB and to the right of BBB.",
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('AAA', object_1).replace('BBB', object_2)

def object1_upleft_object2(object_1, object_2):
    template_candidates = [
        "BBB is to the upper left of AAA.",
        "BBB is to the upper left of AAA.",
        "The object BBB is upper and slightly to the left of the object AAA.",
        "BBB is on the left side and above AAA.",
        "BBB is slightly off center to the top left and AAA is slightly off center to the bottom right.",
        "BBB is positioned in the top left corner of AAA.",
        "BBB is upper left of AAA.",
        "BBB is to the top-left of AAA.",
        "BBB is above AAA at 10 o'clock.",
        "BBB is positioned above and to the left of AAA.",
        "The object BBB is positioned above and to the left of object AAA.",
        "BBB is diagonally left and above AAA.", #BBB is diagonally left and above BBB. - changed by Navdeep
        "BBB is placed at the upper left of AAA.",
        "BBB is sitting at the upper left position to AAA.",
        "AAA is there and BBB is at the 10 position of a clock face.",
        "BBB is to the left and above AAA at an angle of about 45 degrees.", #BBB is to the right and above AAA at an angle of about 45 degrees.,-changed by Navdeep
        "BBB is north west of AAA.",
        "BBB is above and to the left of AAA.",
        "The object AAA and BBB are there. The object BBB is above and slightly to the left of the object AAA.",
        "BBB is directly north west of AAA.",
        "BBB is positioned above AAA and to the left.",
        "BBB is at a 45 degree angle to AAA, in the upper lefthand corner.",
        "BBB is diagonally above AAA to the left at a 45 degree angle.",
        "Object BBB is above object AAA and to the left of it, too.",
        "BBB is diagonally to the upper left of AAA.",
        "BBB presents upper left to AAA.",
        "If BBB is the center of a clock face, AAA is located between 4 and 5.",
        "BBB is above AAA and to the left of AAA.",
    ]
    tmp = random.choice(template_candidates)
    return tmp.replace('BBB', object_1).replace('AAA', object_2)
    

