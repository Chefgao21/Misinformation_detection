import json
import pandas as pd

print("Converting AVeriTeC to RAG format")

# Load the AVeriTeC JSON file
with open('AVeriTeC_train.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} items\n")

articles = []

for i, item in enumerate(data):
    if i >= 10000:  # Limit to 10K
        break
    
    # Extract fields
    claim = item['claim']
    label = item['label']  # Supported, Refuted, Not Enough Evidence
    justification = item.get('justification', '')
    
    # Build article text from questions and answers
    article_text = f"Claim: {claim}\n\n"
    article_text += f"Verdict: {label}\n\n"
    article_text += f"Justification: {justification}\n\n"
    
    # Add Q&A evidence
    if 'questions' in item and item['questions']:
        article_text += "Evidence:\n\n"
        
        for q_item in item['questions'][:3]:  # Max 3 questions
            question = q_item.get('question', '')
            article_text += f"Q: {question}\n"
            
            if 'answers' in q_item and q_item['answers']:
                for answer_item in q_item['answers'][:1]:  # First answer
                    answer = answer_item.get('answer', '')
                    explanation = answer_item.get('boolean_explanation', '')
                    
                    article_text += f"A: {answer}. {explanation}\n\n"
    
    # Convert label to lowercase with underscores
    verdict = label.lower().replace(' ', '_')
    
    articles.append({
        'id': f'averitec_{i}',
        'title': f"Fact Check: {claim[:80]}..." if len(claim) > 80 else f"Fact Check: {claim}",
        'text': article_text,
        'url': item.get('fact_checking_article', f'https://averitec.com/claim/{i}'),
        'verdict': verdict
    })
    
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1:,} items")

# Save to CSV
df = pd.DataFrame(articles)
df.to_csv('fact_check_articles_averitec.csv', index=False)

print(f"Created: {len(df):,} fact-checking articles")

print(f"\n Verdict distribution:")
print(df['verdict'].value_counts())


print(f" Saved to: fact_check_articles_averitec.csv")
