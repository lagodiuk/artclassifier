package artclassifier.algorithm;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import artclassifier.feature.Feature;
import artclassifier.feature.NumericFeature;
import artclassifier.feature.StringFeature;
import artclassifier.util.Name;
import artclassifier.wikitext.WikiPageCategory;
import artclassifier.wikitext.WikiPageSection;
import artclassifier.wikitext.WikiPageTemplate;
import artclassifier.wikitext.WikiPageTemplateArgument;

// TODO: consider usage of wiki markup parsing stuff - for creating better features
// https://code.google.com/p/gwtwiki/wiki/Mediawiki2HTML

public class ArticleClassifier {

	private static final Logger log = Logger.getLogger(ArticleClassifier.class);

	private static final Feature[] FEATURES = getFeatures();

	private static final Filter[] FILTERS = getFilters();

	private static final int ATTRIBUTES_COUNT = FEATURES.length + 1;

	private Attribute labelAttribute;

	private Classifier classifier;

	// TODO: too many options. Create builder
	public ArticleClassifier(
			List<Article> trainingSetArticles,
			List<Article> validationSetArticles,
			Classifier classifier,
			boolean performCrossValidation) throws Exception {

		this.labelAttribute = this.createLabelAttribute(trainingSetArticles);

		Instances trainingSet = this.createInstancesFromLabeledArticles(trainingSetArticles, "trainingSet");

		this.buildClassifier(trainingSet, classifier);

		String report = this.generateReport(trainingSetArticles, validationSetArticles, performCrossValidation);
		log.info(report);
	}

	public Map<String, Double> classifyWithDistribution(Article article) throws Exception {
		Instances classificationSet = this.createEmptyInstancesSet("classificationSet");
		Instance instance = this.articleToInstance(article);
		instance.setDataset(classificationSet);

		double[] distribution = this.classifier.distributionForInstance(instance);

		Map<String, Double> result = new LinkedHashMap<>();
		for (int i = 0; i < this.labelAttribute.numValues(); i++) {
			String label = this.labelAttribute.value(i);
			result.put(label, distribution[i]);
		}

		return result;
	}

	public String classifySingleBestChoice(Article article) throws Exception {
		Instances classificationSet = this.createEmptyInstancesSet("classificationSet");
		Instance instance = this.articleToInstance(article);
		instance.setDataset(classificationSet);

		double[] distribution = this.classifier.distributionForInstance(instance);
		Integer resultNo = 0;
		Double highestResult = 0.0;

		for (int i = 0; i < distribution.length; i++) {
			if (distribution[i] > highestResult){
				highestResult = distribution[i];
				resultNo = i;
			}
		}

		return this.labelAttribute.value(resultNo);
	}

	private void buildClassifier(Instances trainingSet, Classifier classifier) throws Exception {

		if (FILTERS != null) {

			MultiFilter multiFilter = new MultiFilter();
			multiFilter.setInputFormat(trainingSet);
			multiFilter.setFilters(FILTERS);
			FilteredClassifier filteredClassifier = new FilteredClassifier();
			filteredClassifier.setFilter(multiFilter);
			filteredClassifier.setClassifier(classifier);
			this.classifier = filteredClassifier;

		} else {
			this.classifier = classifier;
		}

		this.classifier.buildClassifier(trainingSet);
	}

	private Instances createInstancesFromLabeledArticles(List<Article> labeledArticles, String datasetName) {

		Instances instances = this.createEmptyInstancesSet(datasetName);

		for (Article article : labeledArticles) {
			Instance instance = this.articleToInstance(article);
			instance.setValue(this.labelAttribute, article.getType());
			instances.add(instance);
		}

		return instances;
	}

	private Instance articleToInstance(Article article) {

		Instance instance = new Instance(ATTRIBUTES_COUNT);

		for (Feature feature : FEATURES) {
			feature.addFeature(article, instance);
		}

		return instance;
	}

	private Instances createEmptyInstancesSet(String datasetName) {

		FastVector attributes = new FastVector(ATTRIBUTES_COUNT);

		for (Feature feature : FEATURES) {
			attributes.addElement(feature.getAttribute());
		}

		attributes.addElement(this.labelAttribute);

		Instances instances = new Instances(datasetName, attributes, 1);
		instances.setClass(this.labelAttribute);
		return instances;
	}

	private Attribute createLabelAttribute(List<Article> labeledArticles) {
		Set<String> labels = new HashSet<>();
		for (Article article : labeledArticles) {
			labels.add(article.getType());
		}

		FastVector labelsVector = new FastVector(labels.size());
		for (String label : labels) {
			labelsVector.addElement(label);
		}

		Attribute attr = new Attribute("tagAttribute", labelsVector);
		return attr;
	}

	private String generateReport(List<Article> trainingSetArticles, List<Article> validationSetArticles, boolean performCrossValidation) throws Exception {
		StringBuilder reportBuilder = new StringBuilder().append("\n");
		reportBuilder.append(this.classifier).append("\n");

		if (performCrossValidation) {
			int foldsNum = 10;
			Instances trainingSet = this.createInstancesFromLabeledArticles(trainingSetArticles, "trainingSet");
			this.doCrossValidation(trainingSet, foldsNum, reportBuilder);
		}

		if (validationSetArticles != null) {
			this.evaluateOnValidationSet(validationSetArticles, reportBuilder);
			reportBuilder.append("\n");
			reportBuilder.append("Training set size: ").append(trainingSetArticles.size()).append("\n");
			reportBuilder.append("Validation set size: ").append(validationSetArticles.size()).append("\n");
		}

		String report = reportBuilder.toString();
		return report;
	}

	private void doCrossValidation(Instances trainingSet, int foldsNum, StringBuilder reportBuilder) throws Exception {
		Evaluation cvEaluation = new Evaluation(trainingSet);
		cvEaluation.crossValidateModel(this.classifier, trainingSet, foldsNum, new Random(1));
		reportBuilder.append("\n").append("Cross-validation").append("\n");
		this.appendToReport(reportBuilder, cvEaluation);
	}

	private void evaluateOnValidationSet(List<Article> validationSetArticles, StringBuilder reportBuilder) throws Exception {
		Instances validationSet = this.createInstancesFromLabeledArticles(validationSetArticles, "validationSet");
		Evaluation evaluation = new Evaluation(validationSet);
		evaluation.evaluateModel(this.classifier, validationSet);
		reportBuilder.append("\n").append("Using evaluation-set").append("\n");
		this.appendToReport(reportBuilder, evaluation);
	}

	private void appendToReport(StringBuilder reportBuilder, Evaluation evaluation) throws Exception {
		reportBuilder.append(evaluation.toSummaryString()).append("\n");
		reportBuilder.append(evaluation.toMatrixString()).append("\n");
		reportBuilder.append("FP\tFN\tF-score\tRecall\tPrecision\tLabel").append("\n");
		for (int i = 0; i < this.labelAttribute.numValues(); i++) {
			String label = this.labelAttribute.value(i);

			reportBuilder.append(String.format("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n",
					evaluation.falsePositiveRate(i),
					evaluation.falseNegativeRate(i),
					evaluation.fMeasure(i),
					evaluation.recall(i),
					evaluation.precision(i),
					label));
		}
	}

	private static Filter[] getFilters() {
		List<Filter> filters = new ArrayList<>();
		for (Feature feature : FEATURES) {
			if (feature.hasFilter()) {
				filters.add(feature.getFilter());
			}
		}

		if (filters.isEmpty()) {
			return null;
		}

		return filters.toArray(new Filter[] {});
	}

	private static Feature[] getFeatures() {

		return new Feature[] {

				new StringFeature("plain") {
					@Override
					protected String calculate(Article article) {
						String plain = article.getWikiPageFeatures().getPlain();
						plain = super.prepare(plain);
						return plain;
					}

					@Override
					public StringToWordVector getFilter() {
						StringToWordVector filter = super.getFilter();
						filter.setAttributeNamePrefix("plain_");
						filter.setTFTransform(true);
						filter.setIDFTransform(true);
						filter.setMinTermFreq(5);
						return filter;
					}
				},

				new StringFeature("title") {
					@Override
					protected String calculate(Article article) {
						String title = article.getWikiPageFeatures().getTitle();
						title = super.prepare(title);
						return title;
					}

					@Override
					public StringToWordVector getFilter() {
						StringToWordVector filter = super.getFilter();
						filter.setAttributeNamePrefix("title_");
						return filter;
					}
				},

				new StringFeature("category") {
					@Override
					protected String calculate(Article article) {
						StringBuilder sb = new StringBuilder();
						for (WikiPageCategory category : article.getWikiPageFeatures().getCategories()) {
							sb.append(category.getTitle()).append(" ");
						}
						String categories = sb.toString();
						categories = super.prepare(categories);
						return categories;
					}

					@Override
					public StringToWordVector getFilter() {
						StringToWordVector filter = super.getFilter();
						filter.setAttributeNamePrefix("category_");
						return filter;
					}
				},

				new StringFeature("summary") {
					@Override
					protected String calculate(Article article) {
						String summary = article.getWikiPageFeatures().getSummary();
						summary = super.prepare(summary);
						return summary;
					}

					@Override
					public StringToWordVector getFilter() {
						StringToWordVector filter = super.getFilter();
						filter.setAttributeNamePrefix("summary_");
						filter.setTFTransform(true);
						filter.setIDFTransform(true);
						return filter;
					}
				},

				new StringFeature("sections") {
					@Override
					protected String calculate(Article article) {
						StringBuilder sb = new StringBuilder();
						for (WikiPageSection section : article.getWikiPageFeatures().getSections()) {
							sb.append(section.getTitle()).append(" ");
						}
						String sections = sb.toString();
						sections = super.prepare(sections);
						return sections;
					}

					@Override
					public StringToWordVector getFilter() {
						StringToWordVector filter = super.getFilter();
						filter.setAttributeNamePrefix("section_");
						return filter;
					}
				},

				new StringFeature("template_names") {
					@Override
					protected String calculate(Article article) {
						StringBuilder sb = new StringBuilder();
						for (WikiPageTemplate template : article.getWikiPageFeatures().getTemplates()) {
							sb.append(template.getName()).append(" ");
						}
						String templateNames = sb.toString();
						templateNames = super.prepare(templateNames);
						return templateNames;
					}

					@Override
					public StringToWordVector getFilter() {
						StringToWordVector filter = super.getFilter();
						filter.setAttributeNamePrefix("template_name_");
						return filter;
					}
				},

				new StringFeature("template_arguments") {
					@Override
					protected String calculate(Article article) {
						StringBuilder sb = new StringBuilder();
						for (WikiPageTemplateArgument templateArgument : article.getWikiPageFeatures().getTemplateArguments()) {
							sb.append(templateArgument.getName()).append(" ");
						}
						String templateArguments = sb.toString();
						templateArguments = super.prepare(templateArguments);
						return templateArguments;
					}

					@Override
					public StringToWordVector getFilter() {
						StringToWordVector filter = super.getFilter();
						filter.setAttributeNamePrefix("template_argument_");
						return filter;
					}
				},

				new NumericFeature("title_words_count") {
					@Override
					protected double calculate(Article article) {
						String title = article.getWikiPageFeatures().getTitle();
						int occurences =
								this.countRegrexp(title, "[^\\s]{3,}");
						return occurences;
					}
				},

				new NumericFeature("title_names_count") {
					@Override
					protected double calculate(Article article) {
						String title = article.getWikiPageFeatures().getTitle();
						String[] parts = title.split("\\P{L}+");
						int count = 0;
						for (String p : parts) {
							if (Name.isName(p)) {
								count++;
							}
						}
						return count;
					}
				},

				new NumericFeature("summary_names_count") {
					@Override
					protected double calculate(Article article) {
						String summary = article.getWikiPageFeatures().getSummary();
						String[] parts = summary.split("\\P{L}+");
						int count = 0;
						for (String p : parts) {
							if (Name.isName(p)) {
								count++;
							}
						}
						return count;
					}
				},
		};
	}
}
