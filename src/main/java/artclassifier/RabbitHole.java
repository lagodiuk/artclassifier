package artclassifier;

import artclassifier.algorithm.Article;
import artclassifier.algorithm.ArticleClassifier;
import artclassifier.algorithm.ArticleClassifierService;
import com.rabbitmq.client.*;
import org.json.JSONObject;

import java.io.IOException;
import java.util.HashMap;

public class RabbitHole {
	public static final String INLET_QUEUE_NAME = "ArtClassifier.article.ready.queue";
	public static final String INLET_ROUTING_KEY = "article.ready";
	public static final String INLET_EXCHANGE_NAME = "test_ex";


	public static final String OUTLET_ROUTING_KEY = "article_type.ready";
	public static final String OUTLET_EXCHANGE_NAME = "test_ex";

	public static final String FAILURES_EXCHANGE_NAME = "test_ex";
	public static final String FAILURES_ROUTING_KEY = "ArtClassifier.article.failures";

	public static final String FAILURES_QUEUE_NAME = "ArtClassifier.article.failures.queue";


	public static final String USERNAME;
	public static final String PASSWORD;
	public static final String VIRTUAL_HOST = "events";
	public static final String HOSTNAME;

	static {
		HOSTNAME = System.getenv("RABBIT_HOSTNAME");
		USERNAME = System.getenv("RABBIT_USERNAME");
		PASSWORD = System.getenv("RABBIT_PASSWORD");
	}

	private ConnectionFactory connectionFactory;
	private Channel channel;
	private Connection conn;
	private ArticleClassifier classifier;


	public RabbitHole() {
		connectionFactory = new ConnectionFactory();
		connectionFactory.setUsername(USERNAME);
		connectionFactory.setPassword(PASSWORD);
		connectionFactory.setVirtualHost(VIRTUAL_HOST);
		connectionFactory.setHost(HOSTNAME);
	}

	public ConnectionFactory getConnectionFactory() {
		return connectionFactory;
	}

	public void close() throws IOException {
		conn.close();
		channel.close();
	}

	public void initializeFailuresQueue() throws IOException {
		AMQP.Queue.DeclareOk declareOk = channel.queueDeclare(FAILURES_QUEUE_NAME, true, false, false, null);
		AMQP.Queue.BindOk bindOk = channel.queueBind(declareOk.getQueue(), FAILURES_EXCHANGE_NAME, FAILURES_ROUTING_KEY);
	}

	//FIXME: too generic Exception
	public void init() throws Exception {
		classifier = ArticleClassifierService.getArticleClassifier(false);

		conn = getConnectionFactory().newConnection();
		channel = conn.createChannel();

		HashMap<String, Object> queueArgs = new HashMap<>();
		queueArgs.put("x-dead-letter-exchange", FAILURES_EXCHANGE_NAME);
		queueArgs.put("x-dead-letter-routing-key", FAILURES_ROUTING_KEY);

		AMQP.Queue.DeclareOk declareOk = channel.queueDeclare(INLET_QUEUE_NAME, true, false, false, queueArgs);
		AMQP.Queue.BindOk bindOk = channel.queueBind(declareOk.getQueue(), INLET_EXCHANGE_NAME, INLET_ROUTING_KEY);
	}

	public void launchQueue() throws IOException, InterruptedException {
		QueueingConsumer consumer = new QueueingConsumer(channel);
		channel.basicConsume(INLET_QUEUE_NAME, false, consumer);
		while (true) {
			QueueingConsumer.Delivery delivery = consumer.nextDelivery();

			if (processDelivery(delivery)) {
				channel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);
			} else {
				channel.basicNack(delivery.getEnvelope().getDeliveryTag(), false, false);
			}
		}
	}

	private Boolean processDelivery(QueueingConsumer.Delivery delivery) throws IOException {
		JSONObject obj = new JSONObject(new String(delivery.getBody()));
		if (obj.isNull("title") || obj.isNull("wikitext") || obj.isNull("id") || obj.isNull("wiki_lang")) {
			return false;
		}

		if (!obj.getString("wiki_lang").equalsIgnoreCase("de")){
			return false;
		}

		Article art = new Article();
		art.setTitle(obj.getString("title"));
		art.setWikiText(obj.getString("wikitext"));
		String result;
		try {
			result = classifier.classifySingleBestChoice(art);
			//FIXME: too generic exception
			System.err.println("res: " + result);
		} catch (Exception ex) {
			System.err.println(ex);
			return false;
		}

		JSONObject resultObj = new JSONObject();
		resultObj.put("id", obj.get("id"));
		resultObj.put("article_type", result);
		channel.basicPublish(OUTLET_EXCHANGE_NAME, OUTLET_ROUTING_KEY, null, resultObj.toString().getBytes());
		return true;
	}

	public static void main(String[] args) throws Exception {
		RabbitHole rabbit = new RabbitHole();
		rabbit.init();
		rabbit.initializeFailuresQueue();
		System.err.println("Launching...");
		rabbit.launchQueue();

		rabbit.close();
	}
}