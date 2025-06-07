# Benchmark 용으로 재구성: disease_list 삭제

import dspy

# Omission: Nodule
class DistortReport_Omission_Nodule(dspy.Signature):
    """
    Task: 
        You are given a medical report. Your job is to remove 1 abnormal finding from the report.

    Instructions:
        1. Identify abnormal findings and remove one finding that is directly relevant to 'nodule' or 'mass'.
            - Do not delete findings that do not explicitly contain the keywords.
            - Do not delete findings that indicate normal or not observed. (e.g., "no nodule in both lungs", "nodule resorbed")
            - If no disease is identified, leave the report unchanged and make no modifications.
        
        2. Rules for removing a finding:
            - Deletion must be performed on a sentence basis.
            - Delete only 1 finding from the report. Do not delete more than 1 finding even if there are multiple findings describing 'nodule' or 'mass'.
            - If the selected abnormal finding is the only one in the report, delete it entirely and replace the report with "No relevant findings."
            - In cases where a single sentence contains multiple abnormal findings that are not 'nodule' or 'mass', remove only the portion referring to 'nodule' or 'mass', preserving the rest of the sentence.
            - Do not modify any other parts of the report aside from the deletion of 'nodule' or 'mass'.
            - Ensure that the distorted report remains grammatically and contextually coherent.
       
        3. Define a "finding" as a distinct sentence or phrase describing abnormalities in a specific location.   
        
        4. Do not output any reasoning or explanation.

    Examples:
        Input:
        1. Tiny nodules in left upper lung and right lower lung.
        Output:
        No relevant findings.
        
        Input: 
        Emphysema is noted with multiple bullae and blebs in both upper lungs. 
        A lobulated irregular mass is located in the right upper lung.
        Output: 
        Emphysema is noted with multiple bullae and blebs in both upper lungs.
        
        Input:  
        Interstitial thickening and 2.1cm cavitary nodules in right lung and left lower lung.
        Output:
        Interstitial thickening in right lung and left lower lung.
        
        Input: (Input contains multiple findings, so only 1 finding should be removed)
        Calcified parenchymal nodules of 13 mm and 10 mm in diameter were observed in the anterior segment of the left lung upper lobe. 
        Again, a subpleural, irregularly contoured, 8.5 mm diameter calcified parenchymal nodule was observed in the right lung lower lobe laterobasal segment.
        Output:
        Calcified parenchymal nodules of 13 mm and 10 mm in diameter were observed in the anterior segment of the left lung upper lobe.
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")
    deleted_finding: str = dspy.OutputField(desc="If a finding was deleted from the original report, return the exact part that was deleted. If no finding was deleted, return 0.")
    
# Omission: Effusion
class DistortReport_Omission_Effusion(dspy.Signature):
    """
    Task: 
        You are given a medical report. Your job is to remove 1 abnormal finding from the report.

    Instructions:
        1. Identify abnormal findings and remove one finding that is directly relevant to 'Effusion'.
            - Do not delete findings that do not explicitly contain the keywords.
            - Do not delete findings that indicate normal or not observed. (e.g., "Pleural effusion was not detected")
            - If no disease is identified, leave the report unchanged and make no modifications.
        
        4. Rules for removing a finding:
            - Deletion must be performed on a sentence basis.
            - Delete only 1 finding from the report. Do not delete more than 1 finding even if there are multiple findings describing 'Effusion'.
            - If the selected abnormal finding is the only one in the report, delete it entirely and replace the report with "No relevant findings."
            - In cases where a single sentence contains multiple abnormal findings that are not the 'Effusion', remove only the portion referring to 'Effusion', preserving the rest of the sentence.
            - Do not modify any other parts of the report aside from the deletion of the selected finding.
            - Ensure that the distorted report remains grammatically and contextually coherent
       
        5. Define a "finding" as a distinct sentence or phrase describing abnormalities in a specific location.   
        
        6. Do not output any reasoning or explanation.
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")
    deleted_finding: str = dspy.OutputField(desc="If a finding was deleted from the original report, return the exact part that was deleted. If no finding was deleted, return 0.")
   

# Insertion Nodule
class DistortReport_Insertion_Nodule(dspy.Signature):
    """
    Task:
        You are given a medical report. Your task is to insert a new abnormal finding into the report.
    
    Instructions:
        1. The abnormal finding is stored in the variable inserted_sentence.
        2. To avoid any contradictory situations (e.g., adding an abnormal nodule/mass finding in a report that previously had only normal statements), remove any existing normal findings related to nodules/masses before inserting inserted_sentence.
        3. Insert inserted_sentence naturally into the lung description of the report.
        4. Do not modify any other parts of the report except for removing normal nodule/mass findings and inserting the new abnormal finding.
            - Do not replace the existing findings of nodules/masses; instead, keep them as they are and insert new sentences.
            - Deletion should occur only when there is a normal finding stating the absence of nodules/masses, resulting in a contradiction.
        5. Ensure that the inserted sentence follows the original style and tone of the report.
        6. Avoid any language that suggests the information was added later, such as "new finding," "additionally," "also," or "newly identified."
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    insert_sentence: str = dspy.InputField(desc="the sentence that will be inserted into the report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")

# Insertion Effusion
class DistortReport_Insertion_Effusion(dspy.Signature):
    """
    Task:
        You are given a medical report. Your task is to insert a new abnormal finding into the report.

    Instructions:
        1. The abnormal finding is stored in the variable inserted_sentence.
        2. First, check the radiology report to determine whether pleural effusion is present. Only consider cases for insertion if the report explicitly states that pleural effusion is absent. 
        Since there cannot be multiple sentences related to pleural effusion, skip any reports that mention the presence of pleural effusion.
        3. To avoid any contradictory situations (e.g., adding an pleural effusion finding in a report that previously had normal statements, such as no pleural effusion), remove any existing normal findings related to pleural effusion before inserting inserted_sentence.
        4. Insert inserted_sentence naturally into the the report.
        5. Do not modify any other parts of the report except for removing normal statements of pleural effusion and inserting the new abnormal finding.
            - Do not replace the existing findings of pleural effusion; instead, keep them as they are and insert new sentences.
            - Deletion should occur only when there is a normal finding stating the absence of pleural effusion, resulting in a contradiction.
        6. Ensure that the inserted sentence follows the original style and tone of the report.
        7. Avoid any language that suggests the information was added later, such as "new finding," "additionally," "also," or "newly identified."
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    insert_sentence: str = dspy.InputField(desc="the sentence that will be inserted into the report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")


# Direction: Nodule
class DistortReport_Direction_Nodule(dspy.Signature):
    """
    Task:
        You are given a medical report. Your job is to modify the provided medical report by introducing 1 directional error.

    Instructions:
        1. Identify sentences that include directional terms (e.g., "right," "left," "upper," "lower," "unilateral," "bilateral", "both").
            - Do not modify terms that indicate normal or not observed. (e.g., Do not modify "No mass or nodule in both lungs" to "No mass or nodule in right lung".)
            - If no such terms are identified, leave the report unchanged.
        
        2. From the identified sentences, extract only those that explicitly mention the word "nodule" or "mass". (e.g., "pleural thickening" is not "pleural effusion")
            - Identify variations such as singular/plural forms (e.g., "nodule" and "nodules", "mass" and "masses").
            - Recognize descriptive terms attached to diseases (e.g., "small nodule", "enlarged mass").
        
        3. Randomly select only 1 directional term from the extracted findings and change the directional term, leaving all the others unchanged.
            - Do not change terms that are not directly relected to "nodule" or "mass" even if they contain directional terms.
            - If multiple directional terms exist in the report, modify only 1 term.
            - For findings without directional terms, do not specify a location and leave the finding unchanged.
        
        4. Do not output any reasoning or explanation.
        
        For directional terms describing the lobes of the lung (Left upper lobe, Left lower lobe, Right upper lobe, Right middle lobe, Right lower lobe):
            - Change the term to another within this category (e.g., change "Right upper lobe" to "Left upper lobe" or "Right middle lobe").
        
        For directional terms other than describing the lobes of the lung:
            - Change "left" to "right" and vice versa.
            - Change "upper" to "lower" and vice versa.
            - Change "unilateral" ("right", "left") to "bilateral" ("both") and vice versa.
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")
    distorted_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact modified version of the sentence. If no sentence was modified, return 0.")
    corrected_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact original sentence before modification. If no sentence was modified, return 0.")

# Direction: Effusion
class DistortReport_Direction_Effusion(dspy.Signature):
    """
    Task:
        You are given a medical report. Your job is to modify the provided medical report by introducing 1 directional error.

    Instructions:
        1. Identify sentences that include directional terms (e.g., "right," "left," "upper," "lower," "unilateral," "bilateral", "both").
            - Do not modify terms that indicate normal or not observed. (e.g., Do not modify "No effusion in both lungs" to "No effusion in right lung".)
            - If no such terms are identified, leave the report unchanged.
        
        2. From the identified sentences, extract only those that explicitly mention the word "effusion". (e.g., "pleural thickening" is not "pleural effusion")
            - Identify variations such as singular/plural forms.
            - Recognize descriptive terms attached to diseases (e.g., "small pleural effusion", "moderate effusion").
        
        3. Randomly select only 1 directional term from the extracted findings and change the directional term, leaving all the others unchanged.
            - Do not change terms that are not directly relected to "effusion" even if they contain directional terms.
            - If multiple directional terms exist in the report, modify only 1 term.
            - For findings without directional terms, do not specify a location and leave the finding unchanged.
        
        4. Do not output any reasoning or explanation.
        
        For directional terms describing the lobes of the lung (Left upper lobe, Left lower lobe, Right upper lobe, Right middle lobe, Right lower lobe):
            - Change the term to another within this category (e.g., change "Right upper lobe" to "Left upper lobe" or "Right middle lobe").
        
        For directional terms other than describing the lobes of the lung:
            - Change "left" to "right" and vice versa.
            - Change "upper" to "lower" and vice versa.
            - Change "unilateral" ("right", "left") to "bilateral" ("both") and vice versa.
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")
    distorted_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact modified version of the sentence. If no sentence was modified, return 0.")
    corrected_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact original sentence before modification. If no sentence was modified, return 0.")


# Size: 전체 (완성)
class DistortReport_Size(dspy.Signature):
    """
    Task:
    You are given a medical report. Your job is to generate an erroneous interpretation sentence in the report by altering the size measurement.

    Instructions:
        1. Identify a sentence in the report that mentions a SPECIFIC NUMERIC size measurement (e.g., "1.8 cm", "3 mm").
        2.  Modify the size measurement value in that sentence so that the new value is at least 50% larger or at least 50% smaller than the original.
            - For example, if the original sentence is "A 1.8 cm well-circumscribed nodule in the right upper lobe," then:
                - For an increase: Use a new size greater than 1.8 cm × 1.5 (i.e., greater than 2.7 cm, e.g., 2.8 cm or more).
                - For a decrease: Use a new size that is less than 50% of 1.8 cm (e.g., less than 0.9 cm).
            - For two-dimensional measurements (e.g., "3×4 mm", "2.1×3.2 cm"), modify both dimensions according to the same rule(3×4 mm" → "5×7 mm" (increased) or "1×2 mm" (decreased))
        3. Ensure that all other components of the sentence—such as descriptive text and lobe location—remain unchanged.
        4. DO NOT MODIFY ANY sentences that use only qualitative size descriptions without explicit numeric values, ONLY modify measurements with explicit numbers like "1.8 cm", "3.8 cm", "5 mm":
        - DO NOT modify terms like "millimetric", "centimetric", "milimeter-sized", "large", "small"
        5. Do not modify any other parts of the original report aside from the measurement value.
        6. Do not output any reasoning or explanation.
        7. Do not add new size measurements that weren't in the original report.
   
    Examples of Original Sentences:
        - A 1.8 cm well-circumscribed nodule in the right upper lobe.
        - A 3.8 cm irregular mass in the left upper lobe.
        - There's a 3x7 mm nodule in both upper lobe.
    
    Examples of Erroneous Transformations:
        - Original: A 1.8 cm well-circumscribed nodule in the right upper lobe.
        Erroneous (increased size): A 2.8 cm well-circumscribed nodule in the right upper lobe.
        - Original: A 3.8 cm irregular mass in the left upper lobe.
        Erroneous (decreased size): A 1.5 cm irregular mass in the left upper lobe.
        - Original : There's a 3x7 mm nodule in both upper lobe.
        Erroneous (increased size): There's a 6x10 mm nodule in both upper lobe.
    
    Note:
        - Only the size measurement that includes an exact numeric value and unit (e.g., "5 cm", "2.3 mm") should be altered; all other details must remain exactly as in the original sentence.
        - The new size value must reflect a change of at least 50% compared to the original size.
        - Only 1 lesion's size measurement should be modified.
    """    
    report: str = dspy.InputField(desc="the patient's medical report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")
    
    disease_selected: str = dspy.OutputField(desc="Return the disease that was selected from disease list (e.g., Nodule or Mass, Effusion), return 0 if you did not modify the report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")
    distorted_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact modified version of the sentence. If no sentence was modified, return 0.")
    corrected_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact original sentence before modification. If no sentence was modified, return 0.")
 
# Typo: 전체 (완성)
class DistortReport_Typo(dspy.Signature):
    """
    Task: 
        You are given a medical report. Your job is to introduce exactly 1 spelling error into a single medical term within the report.

    Instructions:
        1. Select 1 medical term in the report and introduce exactly 1 spelling error.
            - Do not introduce errors in normal terms that are not related to radiology findings.
        2. Randomly replace 1 letter in the chosen medical term with a similar-looking character
        3. Introduce additional errors by substituting or adding letters from nearby keyboard keys.
        4. Blank spaces, punctuation, and numbers must remain unchanged.
        5. Do not insert, remove, or modify any other word or phrase. Only introduce a spelling error.
        6. The final output must contain exactly 1 spelling error in a single word and no additional text.
        7. Do not output any reasoning or explanation.
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")

    # disease_selected: str = dspy.OutputField(desc="Return the disease that was distorted, return 0 if you did not modify the report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")
    distorted_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact modified version of the sentence. If no sentence was modified, return 0.")
    corrected_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact original sentence before modification. If no sentence was modified, return 0.")

# Unit: 전체 (완성)
class DistortReport_Unit(dspy.Signature):
    """
    Task: 
        You are given a medical report. Your job is to modify the provided medical report by introducing unit errors.

    Instructions:
        1. Check if the report includes measurement units. If no units are provided (common in over 80% of reports), do not modify the report and skip all subsequent instructions entirely.
        2. If units are present, introduce 1 error by replacing a single measurement unit with an alternative (e.g., cm → mm, mm → m), ensuring only 1 error is introduced per report (error_count=1).
        3. Keep the numerical values unchanged during the modification.
        4. Avoid directly replicating examples or closely imitating provided modifications. Use varied and original modifications while maintaining medical context.
        5. Only modify measurement units if they are explicitly stated in the report. Do not introduce units where none are present.
        6. Do not output any reasoning or explanation.
    """
    report: str = dspy.InputField(desc="the patient's medical report")
    
    distorted_report: str = dspy.OutputField(desc="this is a distorted medical report")

    disease_selected: str = dspy.OutputField(desc="Return the disease that was distorted from disease list, return 0 if you did not modify the report")
    classification: str = dspy.OutputField(desc="Return 1 if you modified the report, return 0 if you did not modify the report")
    distorted_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact modified version of the sentence. If no sentence was modified, return 0.")
    corrected_sentence: str = dspy.OutputField(desc="If you modified any sentence from the original report, return the exact original sentence before modification. If no sentence was modified, return 0.")

# Shuffle 1: 아예 report 바꿔치기 코딩으로
# Shuffle 2: 정상인 환자와 비정상인 환자 바꿔칙 코딩으로