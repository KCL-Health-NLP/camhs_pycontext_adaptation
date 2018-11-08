#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pyConTextNLP.itemData as itemData
import pyConTextNLP.pyConTextGraph as pyConText
import sys
import csv
#sys.path.append("/home/gkotsis/projects/ehostit")
#import sentenceNLP
import nltk
import itertools
import spacy
nlp = spacy.load('en_core_web_sm') 
from collections import Counter
import pandas as pd
import re


#def resolveSentenceNegEx(sentence):
#	from negex import *
#	sentence = sentenceNLP.preprocess(sentence, "suicide")
#	keyword = None
#	import textblob
#	t = textblob.TextBlob(sentence)
#	words = t.words
#	for word in words:
#		if word.startswith('suicid'):
#			keyword = word
#			break
#	if keyword is None:
#		return None
#	rfile = open(r'/home/gkotsis/projects/negex/negex_triggers.txt')
#	irules = sortRules(rfile.readlines())
#	tagger = negTagger(sentence = sentence, phrases = [keyword], rules = irules, negP=False)
#	rs = tagger.getNegationFlag()
#	# print rs
#	if rs=='negated':
#		return False
#	elif rs=='affirmed':
#		return True
#	return rs


## function to map list of pycontext predictions into one document label
def mapPyConTextLabelsToAnnotations(row):
	if 'Positive' in row:
		return 'document_level_suicidal'
	elif 'Negated' in row:
		return 'document_level_nonsuicidal'
	else:
		return 'non_relevant_document'

def mapPyConTextLabelsToAnnotationsMajority(row):
	from collections import Counter
	c = Counter(row)
	nop = c['Positive']
	non = c['Negated']
	if nop==0 and non==0:
		return 'non_relevant_document'
	elif nop>=non:
		return 'document_level_suicidal'
	else:
		return 'document_level_nonsuicidal'


## this function runs pyConText with specified modifier and target keyword files on a dataframe,
## and saves results in two new columns (context_triggers and updated_context)
def resolveAllPyConTextNLP(df, 
	modlexicon="/home/gkotsis/TSUM/sophie_epstein/pycontext/multilingual_lexicon-en-de-fr-sv_v2.csv", 
	targetlexicon = "/home/gkotsis/TSUM/sophie_epstein/pycontext/targets.csv",
	removeFormSentences=False,
	tagExperiencer=False):
	import time
	start_time = time.time()

	## moved reading in lexicons to here

	modifiers = convertCSVtoitemData(modlexicon)
	targets = convertCSVtoitemData(targetlexicon, delimiter=',')
	print('Running pycontext with: ')
	print('Modifiers: '+modlexicon)
	print('Targets: '+targetlexicon)
	#df[[colname1,colname2]] = df['text'].apply(resolveDocumentPyConTextNLP, args=(modlexicon,targetlexicon))
##	##df[['context_triggers', 'updated_context']] = df['text'].apply(resolveDocumentPyConTextNLP, args=(modlexicon,targetlexicon,removeFormSentences,tagExperiencer))
	df[['context_triggers', 'updated_context']] = df['text'].apply(resolveDocumentPyConTextNLP, args=(modifiers,targets,removeFormSentences,tagExperiencer))
	df['mapped_updated_pycontext_1p']=df['updated_context'].apply(mapPyConTextLabelsToAnnotations)
	df['mapped_updated_pycontext_maj']=df['updated_context'].apply(mapPyConTextLabelsToAnnotationsMajority)
	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))
	return df

#def resolveDocumentPyConTextNLP(doc, modlexicon, targetlexicon, removeFormSentences=False, tagExperiencer=False):
def resolveDocumentPyConTextNLP(doc, modifiers, targets, removeFormSentences=False, tagExperiencer=False):
	if doc is not None:
		#print type(doc)
		try:
			doc = doc.encode('utf-8', 'ignore')
			doc = filter(lambda c:ord(c)<128, doc)
		except:
			print 'encode error with doc: '
			#+doc
			print 'using filter function to remove problematic characters'
			doc = filter(lambda c:ord(c)<128, doc)
			#sys.exit(0)
		try:
			doc = doc.decode('utf-8', 'ignore')
		except:
			print 'decode error with doc: '+doc
			sys.exit(0)
		try:
			doc = nlp(unicode(doc))
		except:
			print 'unicode() error with doc: '+doc
			sys.exit(0)
		sents = [sent.string.strip() for sent in doc.sents]
		#print sents
		if removeFormSentences:
			#print 'filtering out sentences from Andreas list'
			#print str(len(sents))
			l = getAndreasList()
			sents = [s for s in sents if not checkIfSentenceInAndreasList(l, s)]
		#print str(len(sents))
	#print [resolveSentencePyConTextNLP(sentence, modlexicon, targetlexicon) for sentence in sents]
##		##pred = [resolveSentencePyConTextNLP(sentence, modlexicon, targetlexicon, tagExperiencer) for sentence in sents]
		pred = [resolveSentencePyConTextNLP(sentence, modifiers, targets, tagExperiencer) for sentence in sents]
		#print len(pred)
		tmp1 = []
		tmp2 = []
		for p in pred:
			if len(p[0]) > 0:
				tmp1.append(p[0])
				trigger = str(p[1][0])
				#print "Trigger: "+trigger
				trigger = trigger.split('phrase')[1]
				trigger = trigger[2:]
				trigger = trigger[0:-2]
				#print "Trigger2: "+trigger
				tmp2.append(trigger)
	#if len(pred) != 2:
	#	return [], []
	#else:
	#	return list(itertools.chain(*pred[0])), list(itertools.chain(*pred[1]))
		#print tmp1, 
		tmp1 = list(itertools.chain(*tmp1))
		#tmp2 = list(itertools.chain(*tmp2))
		#return tmp1, tmp2
		return pd.Series({'updated_context':tmp1, 'context_triggers':tmp2})
	return pd.Series({'updated_context':[], 'context_triggers':[]})

#def resolveSentencePyConTextNLP(sentence, modlexicon, targetlexicon, tagExperiencer=False):
def resolveSentencePyConTextNLP(sentence, modifiers, targets, tagExperiencer=False):
	def getNegationValue(g, te):
		#print type(te)
		if g.isModifiedByCategory(te, "DEFINITE_NEGATED_EXISTENCE"):
			return 'Negated', te
		return 'Positive', te
	def getExperiencerValue(g, te):
		if g.isModifiedByCategory(te, "experiencer"):
			return 'Other', te
		return 'Patient', te
	#Read in modifiers from lexicon file#
	#modlexicon = "/home/gkotsis/TSUM/sophie_epstein/pycontext/lexical_kb_04292013.tsv"
	#modlexicon = "/home/gkotsis/TSUM/sophie_epstein/pycontext/multilingual_lexicon-en-de-fr-sv_v2.csv"
	#modlexicon = "/home/gkotsis/projects/pycontext/negex_orig_triggers_in_pycontext_format.csv"
	#modlexicon = "/home/gkotsis/TSUM/sophie_epstein/pycontext/kcl_negation_cues.csv"

	#create itemData instances from the lexicon#
##	#modifiers = convertCSVtoitemData(modlexicon)

	#targetlexicon = "/home/gkotsis/TSUM/sophie_epstein/pycontext/targets.csv"
	#targetlexicon = "/home/gkotsis/TSUM/sophie_epstein/pycontext/targets_updated.csv"
##	#targets = convertCSVtoitemData(targetlexicon, delimiter=',')

	##TODO:
	# pass whole document
	# parse with spacy
	# implement heuristic
	# import spacy
	# nlp = spacy.load('en_core_web_sm')   
	# doc = WHOLE DOC STRING IN UNICODE
	# sents = [sent.strin.strip() for sent in doc.sents]

	#Define the targets#
	#[ 'suicide','kill herself', 'kill himself', 'kill themselves', 'kill myself',
#	'take his own life', 'take her own life', 'take their own life',
#	'end his own life', 'end her own life', 'end their own life', 'want to die', 
#				'were dead']
#	targets = itemData.itemData()
#	tmp = ['suicide','SUICIDE',r'''suicid*''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['kill herself','SUICIDE',r'''kill.*? herself''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['kill himself','SUICIDE',r'''kill.*? himself''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['kill themselves','SUICIDE',r'''kill.*? themsel[f|ves]''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['kill myself','SUICIDE',r'''kill.*? myself''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['takes her own life','SUICIDE',r'''tak.*? her own life''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['takes his own life','SUICIDE',r'''tak.*? his own life''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['takes their own life','SUICIDE',r'''tak.*? their own life''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['end her life','SUICIDE',r'''end.*?her.*?life''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['end his own life','SUICIDE',r'''end.*? his own life''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['end their own life','SUICIDE',r'''end.*? their own life''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)
#	tmp = ['want to die','SUICIDE',r'''want.*? to die''','']
#	item = itemData.contextItem(tmp)
#	targets.append(item)


	#apply pyConTextNLP on the sentences along with the specified targets and modifiers#
		
	def analyzeSentence(sentence, targets=targets, modifiers=modifiers, tagExperiencer=False):
		#sentence = sentenceNLP.preprocess(sentence, "suicide")
		counter = 0
		counter+=1
		# print "sentence no: "+str(counter)+" - "+sentence
		context = pyConText.ConTextDocument()   
		markup = pyConText.ConTextMarkup()
		markup.setRawText(sentence) 
		markup.markItems(modifiers, mode="modifier")
		markup.markItems(targets, mode="target")
		print markup.getConTextModeNodes('modifier')

		markup.pruneMarks()
		markup.dropMarks('Exclusion')
		markup.applyModifiers()
			
		markup.dropInactiveModifiers()
		markup.updateScopes()
			
		context.addMarkup(markup)
		g = context.getDocumentGraph()
		#print "graph: ",g
		ma = g.getMarkedTargets()
		# if len(ma)==0:
		# 	print sentence
		tst = []
		details = []
		found = {}
		for te in ma:
			#print ma
			tmp1, tmp2 = getNegationValue(g, te)			
			if tagExperiencer:
				e1, e2 = getExperiencerValue(g, te)
				if e1 != 'Other':
					#print e1
					#print sentence
					tst.append(tmp1)
					details.append(tmp2)
					found[tmp2]=Counter(tmp1)
			else:
				tst.append(tmp1)
				details.append(tmp2)
				found[tmp2]=Counter(tmp1)
			#print tmp1, tmp2
			#print e1, e2

		#print tst, details
		return tst, details
	return analyzeSentence(sentence, targets = targets, modifiers=modifiers, tagExperiencer=tagExperiencer)

## added new function to include other modifier types and more targets
def resolveSentencePyConTextNLPExtended(sentence):
	def getNegationValue(g, te):
		hist = getHistoricityValue(g, te)
		if (g.isModifiedByCategory(te, "DEFINITE_NEGATED_EXISTENCE") or g.isModifiedByCategory(te, "PROBABLE_NEGATED_EXISTENCE")) and hist:
			return False
		return True
	def getHistoricityValue(g, te):
		if g.isModifiedByCategory(te, "HISTORICAL"):
			return False
		return True
	def getExperiencerValue(g, te):
		if g.isModifiedByCategory(te, "EXPERIENCER"):
			return False
		return True
	#Read in modifiers from lexicon file#
	modlexicon = "/home/gkotsis/projects/pycontext/lexical_kb_04292013.tsv"
	#modlexicon = "/home/gkotsis/projects/pycontext/negex_orig_triggers_in_pycontext_format.csv"
	# modlexicon = "/home/gkotsis/projects/pycontext/kcl_negation_cues.csv"
	tlexicon = "/home/gkotsis/projects/pycontext/targets_suicidality.csv"

	#create itemData instances from the lexicon#
	modifiers = itemData.instantiateFromCSVtoitemData(modlexicon, 'utf-8', 1, 0, 1, 2, 3)

	#Define the targets#
	#targets = itemData.itemData()
	#tmp = ['suicide','SUICIDE',r'''suicid*''','']
	#item = itemData.contextItem(tmp)
	#targets.append(item)
	targets = itemData.instantiateFromCSVtoitemData(tlexicon, 'utf-8', 1, 0, 1, 2, 3)
	#apply pyConTextNLP on the sentences along with the specified targets and modifiers#
		
	def analyzeSentence(sentence, targets=targets, modifiers=modifiers):
		sentence = sentenceNLP.preprocess(sentence, "suicide")
		print sentence
		counter = 0
		counter+=1
		# print "sentence no: "+str(counter)+" - "+sentence
		context = pyConText.ConTextDocument()   
		markup = pyConText.ConTextMarkup()
		markup.setRawText(sentence) 
		markup.markItems(modifiers, mode="modifier")
		markup.markItems(targets, mode="target")
		   
		markup.pruneMarks()
		markup.dropMarks('Exclusion')
		markup.applyModifiers()
			
		markup.dropInactiveModifiers()
			
		context.addMarkup(markup)
		g = context.getDocumentGraph()

		ma = g.getMarkedTargets()
		print g
		# if len(ma)==0:
		# 	print sentence
		for te in ma:
			print te
			return getNegationValue(g, te)
			

		return None

	return analyzeSentence(sentence, targets = targets, modifiers=modifiers)

def getAndreasList():
	f = open('/home/gkotsis/TSUM/sophie_epstein/pycontext/andrea_section_filters.txt','r')
	list_of_terms = f.readlines()
	f.close()
	list_of_terms = [unicode(l.lower().strip()) for l in list_of_terms]
	return list_of_terms

def checkIfSentenceInAndreasList(andreas_list,sentence):
	for a in andreas_list:
		if a in sentence:
			print 'removing: '+sentence
			return True
	return False

def convertCSVtoitemData(csvFile, encoding='utf-8', delimiter="\t",headerRows=1,
        literalColumn = 0, categoryColumn = 1, regexColumn = 2, ruleColumn = 3):
    """
    takes a CSV file of itemdata rules and creates a single itemData instance.
    csvFile: name of file to read items from
    encoding: unicode enocidng to use; default = 'utf-8'
    headerRows: number of header rows in file; default = 1
    literalColumn: column from which to read the literal; default = 0
    categoryColumn: column from which to read the category; default = 1
    regexColumn: column from which to read the regular expression: default = 2
    ruleColumn: column from which to read the rule; default = 3
    """
    items = itemData.itemData() # itemData to be returned to the user
    header = []
    #reader, f0 = get_fileobj(csvFile)
    #print csvFile
    f = open(csvFile, 'rU')
    reader = csv.reader(f, delimiter=delimiter)
    # first grab numbe rof specified header rows
    for i in range(headerRows):
        row = next(reader)
        header.append(row)
    # now grab each itemData
    for row in reader:
    	#print row
        tmp = [row[literalColumn], row[categoryColumn],
               row[regexColumn], row[ruleColumn]]
        tmp[2] = r'''{0}'''.format(tmp[2]) # convert the regular expression string into a raw string
        item = itemData.contextItem(tmp)
        items.append(item)
    f.close()
    return items
