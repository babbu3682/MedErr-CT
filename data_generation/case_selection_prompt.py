import dspy


# Step 1. Parse the lung section from the full report
class Extract_LungSection(dspy.Signature):
    """
    Task:
        You are given a full chest CT report. Your job is to extract the lung section of the report.

    Instructions:
        1. Select every sentence that describes the lungs or the pleura.
            - Include findings of the lung parenchyma (e.g., nodule, mass, consolidation, ground-glass opacity, emphysema, atelectasis, fibrotic sequela).
            - Include findings of the pleura and the pleural space (e.g., pleural effusion, pleural thickening, pneumothorax).
            - Include sentences that state normal or negative findings of the lungs or the pleura. (e.g., "No pleural effusion was detected.")
            - Do not include sentences that describe only other regions (e.g., trachea and main bronchi, mediastinum, lymph nodes, heart, pericardium, great vessels, esophagus, upper abdomen, bones).

        2. Rules for extraction:
            - Copy each selected sentence exactly as it is written in the report. Do not paraphrase, summarize, translate, correct or merge sentences.
            - Keep the selected sentences in the order in which they appear in the report.
            - Extraction must be performed on a sentence basis. If a sentence describes both the lungs and another region, copy the whole sentence.
            - If the report contains no sentence about the lungs or the pleura, return "No relevant findings."

        3. Do not output any reasoning or explanation.

    Examples:
        Input:
        Trachea and both main bronchi are open. Heart contour and size are normal. A 5 mm nodule was observed in the right lung upper lobe. Pleural effusion was not detected. The liver is normal.
        Output:
        A 5 mm nodule was observed in the right lung upper lobe. Pleural effusion was not detected.

        Input:
        Heart contour and size are normal. No enlarged lymph nodes were detected. Bone structures in the study area are natural.
        Output:
        No relevant findings.
    """

    report: str = dspy.InputField(desc="the patient's full chest CT report")

    lung_section: str = dspy.OutputField(desc="the sentences of the report that describe the lungs or the pleura, copied exactly")


# Step 2. Label the parsed lung section
class Label_LungSection(dspy.Signature):
    """
    Task:
        You are given the lung section of a chest CT report. Your job is to label whether a lung nodule and a pleural effusion are present, and to list the abnormal findings.

    Instructions:
        1. lung_nodule:
            - Return 1 if the report states that at least one nodule is present in the lungs (e.g., "nodule", "nodules", "micronodule", "nodular lesion").
            - Return 0 if no nodule is mentioned, or if a nodule is mentioned only as absent, not observed or resolved. (e.g., "no nodule in both lungs", "nodule resorbed")

        2. pleural_effusion:
            - Return 1 if the report states that a pleural effusion is present on at least one side.
            - Return 0 if no pleural effusion is mentioned, or if a pleural effusion is mentioned only as absent, not observed or resolved. (e.g., "Pleural effusion was not detected.")
            - A pericardial effusion is not a pleural effusion.

        3. abnormality:
            - List every abnormal finding that the report states as present, as short phrases separated by commas.
            - If the report states no abnormal finding, return exactly "no findings".

        4. Use only what is written in the report. Do not infer findings that are not stated.

        5. Do not output any reasoning or explanation.

    Examples:
        Input:
        A 5 mm nodule was observed in the right lung upper lobe. Pleural effusion was not detected.
        Output:
        lung_nodule: 1
        pleural_effusion: 0
        abnormality: nodule in the right lung upper lobe

        Input:
        Bilateral pleural effusion is observed, more prominent on the right. Compressive atelectasis is noted in the adjacent lung parenchyma. No nodule or mass was detected.
        Output:
        lung_nodule: 0
        pleural_effusion: 1
        abnormality: bilateral pleural effusion, compressive atelectasis

        Input:
        No mass, nodule or infiltration was detected in both lung parenchyma. Pleural effusion-thickening was not detected.
        Output:
        lung_nodule: 0
        pleural_effusion: 0
        abnormality: no findings
    """

    report: str = dspy.InputField(desc="the lung section of the patient's chest CT report")

    lung_nodule: str = dspy.OutputField(desc="Return 1 if a lung nodule is present, return 0 otherwise")
    pleural_effusion: str = dspy.OutputField(desc="Return 1 if a pleural effusion is present, return 0 otherwise")
    abnormality: str = dspy.OutputField(desc="comma-separated list of the abnormal findings that are present. If there is none, return exactly 'no findings'")
