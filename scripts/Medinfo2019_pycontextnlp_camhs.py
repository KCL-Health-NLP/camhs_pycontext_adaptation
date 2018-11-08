#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pyConTextNLP.itemData as itemData
import pyConTextNLP.pyConTextGraph as pyConText
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import csv
#sys.path.append("/home/gkotsis/projects/ehostit")
#import sentenceNLP
import nltk
import itertools
import spacy
nlp = spacy.load('en_core_web_sm') 
from collections import Counter
from collections import defaultdict
import pandas as pd
import re
sys.path.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_scripts')
import pyConTextwrapper as pycontextw
import os

def run_configurations(f):
	## dataframe with all annotations:

	df = pd.read_pickle(f)
	#df = df.sample(50)
	## gold annotations in column 'updatedannotationclass'

	## different versions of modifier lexicons:
	modifier_files = []
	## version used for ASD study
	modifier_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_modifiers_AMIA2017.csv')
	## Andrea's terms
	modifier_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_modifiers_SREP2018.csv')	
	## original version of multilingual context (?)
	modifier_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_modifiers_MEDINFO2013.csv')
	## modified versions of Multilingual context
	# old file name: /home/gkotsis/TSUM/sophie_epstein/pycontext/multilingual_lexicon-en-de-fr-sv_v2.csv
	modifier_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_modifiers_MEDINFO2013_v2.csv')
	# old file name: /home/gkotsis/TSUM/sophie_epstein/pycontext/multilingual_lexicon-en-de-fr-sv_v3.csv
	modifier_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_modifiers_MEDINFO2013_v3.csv')
	# old file name: /home/gkotsis/TSUM/sophie_epstein/pycontext/modifier_lexicon_adapted_for_sophie_cohort.csv
	modifier_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_modifiers_MEDINFO2013_v4.csv')
	# old file name: /home/gkotsis/TSUM/sophie_epstein/pycontext/modifier_lexicon_adapted_for_sophie_cohort_ext_conj.csv
	modifier_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_modifiers_MEDINFO2013_v5.csv')


	## target lexicons
	target_files = []
	target_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_targets_minimal_baseline.csv')
	target_files.append('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_targets.csv')

	results = defaultdict(str)
	for t in target_files:
		target_file = t
		tn = target_file.replace('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_','')
		print tn
		for m in modifier_files:
			mn = m.replace('/home/gkotsis/TSUM/sophie_epstein/Medinfo2019/Medinfo2019_lexicons/Medinfo2019_','')
			print mn
			## run pycontext on df -- adds four new columns to df:
			## 'context_triggers': list of triggers that were found by the algorithm 
			## 'updated_context': list of assessments (Positive or Negative)
			## 'mapped_updated_pycontext_1p': document level assertion (using at least one positive --> document_level_suicidal)
			## 'mapped_updated_pycontext_maj'> document level assertion using majority rule
			df = pycontextw.resolveAllPyConTextNLP(df, modlexicon=m, targetlexicon = target_file,removeFormSentences=False, tagExperiencer=False)
			rs = analyse_results(df)
			results[tn+'_'+mn] = rs
			patients = getPatients(df)
			prs = analyse_results(patients)
			results[tn+'_'+'patient_level_'+mn] = prs
			## including filtering of Andrea's form sentences
			#df = pycontextw.resolveAllPyConTextNLP(df, modlexicon=m, targetlexicon = target_file,removeFormSentences=True, tagExperiencer=False)
			#rs = analyse_results(df)
			#results[tn+'_'+'form_filter_'+mn] = rs
			#patients = getPatients(df)
			#prs = analyse_results(patients)
			#results[tn+'_'+'form_filter_'+'patient_level_'+mn] = prs
			## including test excluding experiencer
			df = pycontextw.resolveAllPyConTextNLP(df, modlexicon=m, targetlexicon = target_file,removeFormSentences=False, tagExperiencer=True)
			rs = analyse_results(df)
			results[tn+'_'+'experiencer_filter_'+mn] = rs
			patients = getPatients(df)
			prs = analyse_results(patients)
			results[tn+'_'+'experiencer_filter_'+'patient_level_'+mn] = prs
	return results, df

def getPatients(df, annotationLabel='updatedannotationclass', predictionLabel='mapped_updated_pycontext_1p'):
	rs = pd.DataFrame()
	for brcid in df['brcid'].unique():
		patient = {}
		patient['brcid'] = brcid
		patient[annotationLabel] = None
		patient[predictionLabel] = None
		if 'document_level_suicidal' in df[df['brcid']==brcid][annotationLabel].tolist():
			patient[annotationLabel] = 'suicidal'
		elif 'document_level_nonsuicidal' in df[df['brcid']==brcid][annotationLabel].tolist():
			patient[annotationLabel] = 'nonsuicidal'
		else:
			patient[annotationLabel] = 'non_relevant'

		if 'document_level_suicidal' in df[df['brcid']==brcid][predictionLabel].tolist():
			patient[predictionLabel] = 'suicidal'
		elif 'document_level_nonsuicidal' in df[df['brcid']==brcid][predictionLabel].tolist():
			patient[predictionLabel] = 'nonsuicidal'
		else:
			patient[predictionLabel] = 'non_relevant'
		tmp = pd.DataFrame(patient.values(), index=patient.keys()).transpose()
		rs = pd.concat([rs, tmp])
	
	return rs

def analyse_results(df, gold_col='updatedannotationclass', pred_col='mapped_updated_pycontext_1p'):
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import classification_report
	rs = classification_report(df[gold_col],df[pred_col])
	#print type(rs)
	#print rs
	return rs

def results_in_printable_format(results):
	res = {}
	for t in results:
		#print t
		r = results[t]
		r = r.split('\n')
		res2 = {}
		for rr in r:
			#print rr
			match = re.search('(.*?)(\d\.\d\d)\s+(\d\.\d\d)\s+(\d\.\d\d)\s+(\d+)', rr)
			if match is not None:
				#print rr
				#res2 = {}
				v = match.group(1).strip()
				p = match.group(2)
				r = match.group(3)
				f = match.group(4)
				s = match.group(5)
				res2[v] = [p,r,f,s]
				res[t] = res2
	return res

def results_to_dataframe(results):
	res = results_in_printable_format(results)
	d_exp = {}
	d = {}
	p_exp = {}
	p = {}
	tmp = []
	h = ['id', 'config_target', 'config_modifier', 'label', 'precision', 'recall', 'fscore', 'support']
	tmp.append(h)
	for r in res:
		if 'patient' in r:
			if 'experiencer' in r:
				p_exp[r] = res[r]
			else:
				p[r] = res[r]
		else:
			if 'experiencer' in r:
				d_exp[r] = res[r]
			else:
				d[r] = res[r]
	c = 0
	for dd in d:
		print dd
		for ddd in d[dd]:
			tmp2 = []
			tmp2.append(c)
			c +=1
			tm = dd.split('csv')
			tmp2.append(tm[0])
			tmp2.append(tm[1])
			tmp2.append(ddd)
			for dddd in d[dd][ddd]:
				tmp2.append(dddd)
			tmp.append(tmp2)
	for dd in d_exp:	
		for ddd in d_exp[dd]:
			tmp2 = []
			c+=1
			tmp2.append(c)
			tm = dd.split('csv')
			tmp2.append(tm[0])
			tmp2.append(tm[1])
			#tmp2.append(dd)
			tmp2.append(ddd)
			for dddd in d_exp[dd][ddd]:
				tmp2.append(dddd)
			tmp.append(tmp2)
	for dd in p:	
		for ddd in p[dd]:
			tmp2 = []
			c+=1
			tmp2.append(c)
			tm = dd.split('csv')
			tmp2.append(tm[0])
			tmp2.append(tm[1])
			#tmp2.append(dd)
			tmp2.append(ddd)
			for dddd in p[dd][ddd]:
				tmp2.append(dddd)
			tmp.append(tmp2)
	for dd in p_exp:	
		for ddd in p_exp[dd]:
			tmp2 = []
			c+=1
			tmp2.append(c)
			#tmp2.append(dd)
			tm = dd.split('csv')
			print tm
			tmp2.append(tm[0])
			tmp2.append(tm[1])
			tmp2.append(ddd)
			for dddd in p_exp[dd][ddd]:
				tmp2.append(dddd)
			tmp.append(tmp2)
	#print tmp
	#print tmp[0]
	df = pd.DataFrame(tmp[1:], columns=tmp[0])
	#print df.head()
	return df


def main_run():

	base_dir = '/home/gkotsis/brc_nlp_sumithra/sophie_epstein/algorithm_evaluation/'
	datasets = []
	results_to_return = {}

	datasets.append(os.path.join(base_dir, 'datasets/all_annotations_training_data_minimal.pickle'))
	datasets.append(os.path.join(base_dir, 'datasets/blind_test_sets_combined_annotations.pickle'))
	for d in datasets:
		new_f = str(d)
		new_f = new_f.split('/')
		new_f = new_f[-1]
		new_f = new_f.replace('pickle', 'xlsx')
		print new_f
		new_f = os.path.join(base_dir, 'results/results_20181023_'+new_f)
		print new_f
		results, df = run_configurations(d)
		resultsdf = results_to_dataframe(results)
		resultsdf.to_excel(os.path.join(base_dir, new_f))
		results_to_return[new_f] = resultsdf
	return results_to_return



#################################################################################
## OLD
#################################################################################

def final_run():

	base_dir = '/home/gkotsis/TSUM/sophie_epstein/'
	datasets = []
	results_to_return = {}

#	datasets.append(os.path.join(base_dir, 'datasets/blind_test_set2_annotations.pickle'))
#	datasets.append(os.path.join(base_dir, 'datasets/blind_test_set1_annotations.pickle'))
#	datasets.append(os.path.join(base_dir, 'datasets/gold_all_predictions.pickle'))
	datasets.append(os.path.join(base_dir, 'datasets/blind_test_sets_combined_annotations.pickle'))
#	datasets.append(os.path.join(base_dir, 'datasets/all_annotated_sets_combined.pickle'))
	for d in datasets:
		new_f = str(d)
		new_f = new_f.split('/')
		new_f = new_f[-1]
		new_f = new_f.replace('pickle', 'xlsx')
		print new_f
		new_f = os.path.join(base_dir, 'results/results_20180702_3_'+new_f)
		print new_f
		results, df = run_final_configurations(d)
		resultsdf = results_to_dataframe(results)
		resultsdf.to_excel(os.path.join(base_dir, new_f))
		results_to_return[new_f] = resultsdf
	return results_to_return, df


def run_final_configurations(f):
	## dataframe with all annotations:

	df = pd.read_pickle(f)
	results = defaultdict(str)
	#df = df.sample(50)
	## gold annotations in column 'updatedannotationclass'

	#####
	## on the combined test sets, the best performing on patient level was:
	#####
	m = '/home/gkotsis/TSUM/sophie_epstein/pycontext/multilingual_lexicon-en-de-fr-sv_v3.csv'
	t = '/home/gkotsis/TSUM/sophie_epstein/pycontext/targets_updated10April2018.csv'

	df = pycontextw.resolveAllPyConTextNLP(df, modlexicon=m, targetlexicon = t,removeFormSentences=False, tagExperiencer=True)

	mn = m.replace('/home/gkotsis/TSUM/sophie_epstein/pycontext/','')
	mn = mn.replace('.csv','')
	print mn

	tn = t.replace('/home/gkotsis/TSUM/sophie_epstein/pycontext/','')
	print tn
	tntmp = tn.replace('.csv','')

	rs = analyse_results(df)
	results[tn+'_'+'experiencer_filter_'+mn] = rs
	patients = getPatients(df)
	prs = analyse_results(patients)
	results[tn+'_'+'experiencer_filter_'+'patient_level_'+mn] = prs
	## rename column and save everything in dataframe
	df = df.rename(columns={'context_triggers':'context_triggers_'+tntmp+'_'+mn, 
				'updated_context':'updated_context_'+tntmp+'_'+mn,
				'mapped_updated_pycontext_1p': 'mapped_updated_pycontext_1p_'+tntmp+'_'+mn,
				'mapped_updated_pycontext_maj':'mapped_updated_pycontext_maj_'+tntmp+'_'+mn})
	
	#####
	## best compromise between patient and document level
	#####
	m = '/home/gkotsis/TSUM/sophie_epstein/pycontext/modifier_lexicon_adapted_for_sophie_cohort_ext_conj.csv'
	t = '/home/gkotsis/TSUM/sophie_epstein/pycontext/targets_updated10April2018.csv'

	df = pycontextw.resolveAllPyConTextNLP(df, modlexicon=m, targetlexicon = t,removeFormSentences=False, tagExperiencer=True)

	mn = m.replace('/home/gkotsis/TSUM/sophie_epstein/pycontext/','')
	mn = mn.replace('.csv','')
	print mn

	tn = t.replace('/home/gkotsis/TSUM/sophie_epstein/pycontext/','')
	print tn
	tntmp = tn.replace('.csv','')

	rs = analyse_results(df)
	results[tn+'_'+'experiencer_filter_'+mn] = rs
	patients = getPatients(df)
	prs = analyse_results(patients)
	results[tn+'_'+'experiencer_filter_'+'patient_level_'+mn] = prs
	## rename column and save everything in dataframe
	df = df.rename(columns={'context_triggers':'context_triggers_'+tntmp+'_'+mn, 
				'updated_context':'updated_context_'+tntmp+'_'+mn,
				'mapped_updated_pycontext_1p': 'mapped_updated_pycontext_1p_'+tntmp+'_'+mn,
				'mapped_updated_pycontext_maj':'mapped_updated_pycontext_maj_'+tntmp+'_'+mn})

	
	return results, df

def run_on_full_cohort(df):
	m = '/home/gkotsis/TSUM/sophie_epstein/pycontext/multilingual_lexicon-en-de-fr-sv_v3.csv'
	t = '/home/gkotsis/TSUM/sophie_epstein/pycontext/targets_updated10April2018.csv'

	df = pycontextw.resolveAllPyConTextNLP(df, modlexicon=m, targetlexicon = t,removeFormSentences=False, tagExperiencer=True)
	rs = df
	rs = rs.drop('text', axis=1)
	rs.to_pickle('/home/gkotsis/brc_nlp_sumithra/sophie_epstein/full_cohort_system_output.pickle')

	return df

def run_alternative_configuration_on_full_cohort(df):
	m = '/home/gkotsis/TSUM/sophie_epstein/pycontext/modifier_lexicon_adapted_for_sophie_cohort_ext_conj.csv'
	t = '/home/gkotsis/TSUM/sophie_epstein/pycontext/targets_updated10April2018.csv'

	df = pycontextw.resolveAllPyConTextNLP(df, modlexicon=m, targetlexicon = t,removeFormSentences=False, tagExperiencer=True)
	rs = df
	rs = rs.drop('text', axis=1)
	rs.to_pickle('/home/gkotsis/brc_nlp_sumithra/sophie_epstein/full_cohort_alternative_system_output.pickle')

	return df

def getGoldPatients(df, annotationLabel='updatedannotationclass'):
	rs = pd.DataFrame()
	for brcid in df['brcid'].unique():
		patient = {}
		patient['brcid'] = brcid
		patient[annotationLabel] = None
		if 'document_level_suicidal' in df[df['brcid']==brcid][annotationLabel].tolist():
			patient[annotationLabel] = 'suicidal'
		elif 'document_level_nonsuicidal' in df[df['brcid']==brcid][annotationLabel].tolist():
			patient[annotationLabel] = 'nonsuicidal'
		else:
			patient[annotationLabel] = 'non_relevant'

		tmp = pd.DataFrame(patient.values(), index=patient.keys()).transpose()
		rs = pd.concat([rs, tmp])
	
	return rs
def getGoldPredPatients(df, annotationLabel='updatedannotationclass', predictionLabel='mapped_updated_pycontext_1p'):
	rs = pd.DataFrame()
	for brcid in df['brcid'].unique():
		patient = {}
		patient['brcid'] = brcid
		patient[annotationLabel] = None
		patient[predictionLabel] = None
		if 'document_level_suicidal' in df[df['brcid']==brcid][annotationLabel].tolist():
			patient[annotationLabel] = 'suicidal'
		elif 'document_level_nonsuicidal' in df[df['brcid']==brcid][annotationLabel].tolist():
			patient[annotationLabel] = 'nonsuicidal'
		else:
			patient[annotationLabel] = 'non_relevant'

		if 'document_level_suicidal' in df[df['brcid']==brcid][predictionLabel].tolist():
			patient[predictionLabel] = 'suicidal'
		elif 'document_level_nonsuicidal' in df[df['brcid']==brcid][predictionLabel].tolist():
			patient[predictionLabel] = 'nonsuicidal'
		else:
			patient[predictionLabel] = 'non_relevant'
		tmp = pd.DataFrame(patient.values(), index=patient.keys()).transpose()
		rs = pd.concat([rs, tmp])
	
	return rs

#if __name__ == '__main__':
#	main()