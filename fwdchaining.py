class Rule:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

    def is_triggered(self, facts):
        return all(fact in facts for fact in self.antecedent)

class KnowledgeBase:
    def __init__(self):
        self.rules = []

    def add_rule(self, antecedent, consequent):
        rule = Rule(antecedent, consequent)
        self.rules.append(rule)

    def forward_chaining(self, facts):
        new_facts = facts.copy()
        while True:
            rule_triggered = False
            for rule in self.rules:
                if rule.is_triggered(new_facts) and rule.consequent not in new_facts:
                    new_facts.append(rule.consequent)
                    rule_triggered = True
            if not rule_triggered:
                break
        return new_facts


kb = KnowledgeBase()

num_rules = int(input("Enter the number of rules: "))
for i in range(num_rules):
    antecedent = input(f"Enter the antecedent of rule {i + 1} (separated by spaces): ").split()
    consequent = input(f"Enter the consequent of rule {i + 1}: ")
    kb.add_rule(antecedent, consequent)

initial_facts = input("Enter the initial facts (separated by spaces): ").split()


final_facts = kb.forward_chaining(initial_facts)
print("Final facts after forward chaining:", final_facts)

# sample output:
# Enter the number of rules:  3
# Enter the antecedent of rule 1 (separated by spaces):  A B
# Enter the consequent of rule 1:  C
# Enter the antecedent of rule 2 (separated by spaces):  B C
# Enter the consequent of rule 2:  D
# Enter the antecedent of rule 3 (separated by spaces):  D
# Enter the consequent of rule 3:  E
# Enter the initial facts (separated by spaces):  A B
# Final facts after forward chaining: ['A', 'B', 'C', 'D', 'E']

