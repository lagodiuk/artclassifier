package artclassifier.util;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class UsefulnessClassifier extends Classifier {

	private static final String USEFUL = "USEFUL";

	private static final String USELESS = "USELESS";

	private Classifier usefulnessClassifier;

	private Classifier normalClassifier;

	private String uselessLabel;

	private List<String> dataLabels = new ArrayList<>();

	public UsefulnessClassifier(Classifier usefulnessClassifier, Classifier normalClassifier, String uselessLabel) {
		this.usefulnessClassifier = usefulnessClassifier;
		this.normalClassifier = normalClassifier;
		this.uselessLabel = uselessLabel;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		this.buildUsefulnessClassifier(data);
		this.buildNormalClassifier(data);
		System.out.println(this.usefulnessClassifier);
		System.out.println(this.normalClassifier);
	}

	private void buildNormalClassifier(Instances data) throws Exception {

		int numAttributes = data.numAttributes();
		FastVector newAttributes = new FastVector(numAttributes);

		FastVector newClassAttributeValues = new FastVector(data.classAttribute().numValues() - 1);
		for (int i = 0; i < data.classAttribute().numValues(); i++) {
			String value = data.classAttribute().value(i);
			this.dataLabels.add(value);
			if (this.uselessLabel.equals(value)) {
				continue;
			}
			newClassAttributeValues.addElement(value);
		}
		Attribute newClassAttribute = new Attribute(data.classAttribute().name(), newClassAttributeValues);

		for (int i = 0; i < numAttributes; i++) {
			Attribute attr = data.attribute(i);
			if (i != data.classIndex()) {
				newAttributes.addElement(attr);
			} else {
				newAttributes.addElement(newClassAttribute);
			}
		}

		Instances newData = new Instances("usefulDataset", newAttributes, 1);
		newData.setClass(newClassAttribute);

		for (int i = 0; i < data.numInstances(); i++) {
			System.out.println(i);
			Instance instance = data.instance(i);
			Instance newInstance = new Instance(numAttributes);
			boolean isUseless = false;
			for (int j = 0; j < numAttributes; j++) {
				Attribute attr = data.attribute(j);
				if (j != data.classIndex()) {
					if (attr.isNumeric()) {
						newInstance.setValue(attr, instance.value(attr));
					} else {
						newInstance.setValue(attr, instance.stringValue(attr));
					}
				} else {
					if (this.uselessLabel.equals(instance.stringValue(attr))) {
						isUseless = true;
					} else {
						newInstance.setValue(newClassAttribute, instance.stringValue(attr));
					}
				}
			}
			if (!isUseless) {
				newData.add(newInstance);
			}
		}

		this.normalClassifier.buildClassifier(newData);
	}

	private void buildUsefulnessClassifier(Instances data) throws Exception {

		int numAttributes = data.numAttributes();
		FastVector newAttributes = new FastVector(numAttributes);

		FastVector newClassAttributeValues = new FastVector(2);
		newClassAttributeValues.addElement(USEFUL);
		newClassAttributeValues.addElement(USELESS);
		Attribute newClassAttribute = new Attribute(data.classAttribute().name(), newClassAttributeValues);
		for (int i = 0; i < numAttributes; i++) {
			Attribute attr = data.attribute(i);
			if (i != data.classIndex()) {
				newAttributes.addElement(attr);
			} else {
				newAttributes.addElement(newClassAttribute);
			}
		}

		Instances newData = new Instances("usefulnessDataset", newAttributes, 1);
		newData.setClass(newClassAttribute);

		for (int i = 0; i < data.numInstances(); i++) {
			System.out.println(i);
			Instance instance = data.instance(i);
			Instance newInstance = new Instance(numAttributes);
			for (int j = 0; j < numAttributes; j++) {
				Attribute attr = data.attribute(j);
				if (j != data.classIndex()) {
					if (attr.isNumeric()) {
						newInstance.setValue(attr, instance.value(attr));
					} else {
						newInstance.setValue(attr, instance.stringValue(attr));
					}
				} else {
					if (this.uselessLabel.equals(instance.stringValue(attr))) {
						newInstance.setValue(newClassAttribute, USELESS);
					} else {
						newInstance.setValue(newClassAttribute, USEFUL);
					}
				}
			}
			newData.add(newInstance);
		}

		this.usefulnessClassifier.buildClassifier(newData);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] dist = new double[instance.numClasses()];

		double[] usefulnessDistr = this.usefulnessClassifier.distributionForInstance(instance);
		if (usefulnessDistr[1] > usefulnessDistr[0]) {
			for (int i = 0; i < this.dataLabels.size(); i++) {
				if (this.dataLabels.get(i).equals(this.uselessLabel)) {
					dist[i] = 1;
					return dist;
				}
			}
		}

		double[] normalDistr = this.normalClassifier.distributionForInstance(instance);
		int j = 0;
		for (int i = 0; i < this.dataLabels.size(); i++) {
			if (this.dataLabels.get(i).equals(this.uselessLabel)) {
				dist[i] = 0;
			} else {
				dist[i] = normalDistr[j++];
			}
		}

		return dist;
	}
}
