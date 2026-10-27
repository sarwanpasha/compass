"""Unrelated HGDP+1kG samples with genetic region and population labels."""
import pandas as pd, json
from common import D
m = pd.read_csv(D + 'hgdp1kg/gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.tsv.bgz', sep='\t', compression='gzip')
rows = []
for s, a, b in zip(m.s, m.hgdp_tgp_meta, m.relatedness_inference):
    a = json.loads(a); b = json.loads(b)
    rows.append((s, a.get('genetic_region'), a.get('population'), a.get('project'), b.get('related')))
d = pd.DataFrame(rows, columns=['s', 'region', 'population', 'project', 'related'])
d = d[d.region.notna() & (d.related == False)]
d.to_csv(D + 'hgdp1kg/samples_unrelated_pop.tsv', sep='\t', index=False)
print(d.region.value_counts()); print(d[d.project == '1000 Genomes'].population.value_counts().head(30))
