from copy_chunking import HierarchicalParser, SAMPLE_DOCUMENT


parser = HierarchicalParser()
chunks = parser.parse(SAMPLE_DOCUMENT)

print(chunks['3_1_đ_nh_ngh_a_hàm_cơ_b_n_70b5'].content)
print(chunks['3_2_tham_s__và_đ_i_s_699a'].content)