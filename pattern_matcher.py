class PatternBasedPromptRecovery:
    def __init__(self):
        self.patterns = {
            'sea_shanty': {
                'keywords': ['verse', 'chorus', 'ahoy', 'sail', 'sea'],
                'template': 'Convert this into a sea shanty: """{original_text}"""'
            },
            'shakespeare': {
                'keywords': ['thee', 'thou', 'hath', 'doth', 'forsooth', 'verily'],
                'template': 'Rewrite this as a Shakespearean sonnet: """{original_text}"""'
            },
            'pirate': {
                'keywords': ['arr', 'matey', 'ye', 'booty', 'treasure', 'ship'],
                'template': 'Transform this into a pirate\'s speech: """{original_text}"""'
            },
            'legal': {
                'keywords': ['hereby', 'pursuant', 'aforementioned', 'whereas', 'shall'],
                'template': 'Rewrite this in the style of a legal document: """{original_text}"""'
            },
            'haiku': {
                'keywords': ['syllables', '5-7-5', 'nature', 'season'],
                'template': 'Rewrite this as a series of haikus: """{original_text}"""'
            },
            'rap': {
                'keywords': ['yo', 'beat', 'rhyme', 'flow', 'mic'],
                'template': 'Convert this into a rap song: """{original_text}"""'
            },
            'fairy_tale': {
                'keywords': ['once upon a time', 'kingdom', 'princess', 'prince', 'magic'],
                'template': 'Transform this into a fairy tale: """{original_text}"""'
            },
            'dialogue': {
                'keywords': [': "', ':" ', 'said', 'asked', 'replied'],
                'template': 'Convert this into a dialogue between two people: """{original_text}"""'
            },
            'bullet_points': {
                'keywords': ['•', '- ', '* '],
                'template': 'Rewrite this as a bullet point list: """{original_text}"""'
            },
            'step_by_step': {
                'keywords': ['step 1', 'first', 'second', 'third', 'finally'],
                'template': 'Transform this into a step-by-step guide: """{original_text}"""'
            }
        }
    
    def predict(self, original_text, rewritten_text):
        """Predict rewrite prompt based on patterns in the rewritten text"""
        scores = {}
        
        # Calculate scores for each pattern
        for pattern_name, pattern_info in self.patterns.items():
            score = 0
            for keyword in pattern_info['keywords']:
                if keyword.lower() in rewritten_text.lower():
                    score += 1
            
            scores[pattern_name] = score / len(pattern_info['keywords'])
        
        # Find the best matching pattern
        best_pattern = max(scores.items(), key=lambda x: x[1])
        
        # If the score is too low, use a generic template
        if best_pattern[1] < 0.2:
            return f'Rewrite the following text: """{original_text}"""'
        
        return self.patterns[best_pattern[0]]['template'].format(original_text=original_text)
