import random
from synthesis.config import PROFESSIONS, SENIORITY_EXP, LOCATIONS

STATUSES = ["internal", "external"]

class CandidatePlanner:
    def __init__(self, target_count: int = 10):
        self.target_count = target_count

    def generate_skeletons(self) -> list:
        skeletons = []
        for i in range(self.target_count):
            profession = random.choice(list(PROFESSIONS.keys()))
            seniority = random.choice(list(SENIORITY_EXP.keys()))
            status = random.choice(STATUSES)
            
            exp_range = SENIORITY_EXP[seniority]
            years_experience = random.randint(exp_range[0], exp_range[1])
            department = PROFESSIONS[profession]
            
            skeleton = {
                "status": status,
                "profession": profession,
                "seniority": seniority,
                "department": department,
                "years_experience": years_experience
            }
            
            if status == "internal":
                max_tenure = max(1.0, float(years_experience))
                skeleton["tenure_years"] = round(random.uniform(0.5, max_tenure), 1)
            else:
                skeleton["location"] = random.choice(LOCATIONS)
                skeleton["salary_expectation"] = random.randint(80, 250) * 1000
                skeleton["notice_period_days"] = random.choice([14, 30, 60, 90])
                
            skeletons.append(skeleton)
        return skeletons
