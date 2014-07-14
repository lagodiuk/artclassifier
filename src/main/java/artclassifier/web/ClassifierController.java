package artclassifier.web;

import java.util.Map;
import java.util.Map.Entry;

import javax.annotation.PostConstruct;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import artclassifier.Article;
import artclassifier.ArticleClassifier;
import artclassifier.ArticleClassifierService;
import artclassifier.wikia.WikiaArticlesExtractor;

@Controller
public class ClassifierController {

	private ArticleClassifier classifier;

	@PostConstruct
	public void init() throws Exception {
		this.classifier = ArticleClassifierService.getArticleClassifier();
	}

	@RequestMapping("/ping")
	@ResponseBody
	public String ping() {
		return "Ok";
	}

	@RequestMapping("/classify")
	@ResponseBody
	public String classify(@RequestParam String url) throws Exception {
		Article article = WikiaArticlesExtractor.getArticleByURL(url);
		this.classifier.classifyWithDistribution(article);
		Map<String, Double> result = this.classifier.classifyWithDistribution(article);

		// Java 8 stuff - must be refactored
		// List<Entry<String, Double>> resultSortedByValue =
		// result.entrySet().stream()
		// .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
		// .collect(Collectors.toList());

		StringBuilder sb = new StringBuilder();

		sb.append("<a href=\"").append(url).append("\">").append(url).append("</a><br/><br/>");

		int i = 0;
		for (Entry<String, Double> entry : result.entrySet()) {
			if (i == 0) {
				sb.append("Classified as article about: ");
				sb.append(entry.getKey()).append("<br/>");
				sb.append("</br>Other classes, sorted by relevance for this article<br/>");
			} else {
				sb.append(i).append(". ").append(entry.getKey()).append("<br/>");
			}
			i++;
		}

		return sb.toString();
	}

}
