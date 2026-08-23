import json
r = json.load(open('text.json'))

# Plan structure
plan = r['plan']
print('=== PLAN ===')
print('type:', plan['question_type'])
print('conditions:', plan.get('conditions', []))
print('populations:', plan.get('populations', []))
print('target:', plan.get('target'))
print('requirements:', len(plan['requirements']))
for req in plan['requirements']:
    print('  ' + req['id'] + ': ' + req['topic'] + ' | pop=' + req['population'] + ' | focus=' + req['focus'] + ' | target=' + req['target'])
print('queries:', len(plan['queries']))
for q in plan['queries'][:8]:
    print('  ' + q['id'] + ' -> ' + str(q['requirement_ids']) + ' : ' + q['text'][:100])

# Evidence types
print()
print('=== EVIDENCE ===')
types = {}
for c in r['final_evidence']:
    types[c['node_type']] = types.get(c['node_type'], 0) + 1
    if c['node_type'].startswith('table'):
        print('  TABLE:', c['chunk_id'], '(' + c['paper_id'] + ')')

print('types:', types)

# Coverage
print()
print('=== COVERAGE ===')
cov = r['coverage']
print('all_covered:', cov['all_requirements_covered'])
print('covered:', cov['covered'])
print('uncovered:', cov['uncovered'])
print('fraction:', cov['coverage_fraction'])

# Selected papers
print()
print('=== SELECTED PAPERS (top 5) ===')
for p in r['papers'][:5]:
    print('  ' + p['paper_id'] + '  score=' + str(round(p['paper_score'],3)) + '  reqs=' + str(p['supported_requirements']) + '  ' + p['reason'])
PY