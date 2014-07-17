package artclassifier;

import com.rabbitmq.client.*;
import org.json.JSONObject;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public class RabbitHole {
	public static final String INLET_QUEUE_NAME = "ArtClassifier.article.ready.queue";
	public static final String INLET_ROUTING_KEY = "article.ready";

	public static final String OUTLET_ROUTING_KEY = "article_type.ready";


	public static final String INLET_EXCHANGE_NAME = "test_ex";
	public static final String FAILURES_EXCHANGE_NAME = "test_ex";
	public static final String FAILURES_ROUTING_KEY = "ArtClassifier.article.failures";


	public static final String USERNAME;
	public static final String PASSWORD;
	public static final String VIRTUAL_HOST = "events";
	public static final String HOSTNAME;

	static {
		HOSTNAME = "dev-datapi-s1";
		USERNAME = "ev-guest";
		PASSWORD = "ev-guest";
	}

	private ConnectionFactory connectionFactory;
	private Boolean objectInitialized = false;
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

	public void close() throws IOException{
		conn.close();
		channel.close();
	}

	//FIXME: too generic Exception
	public void init() throws Exception {
		if (!objectInitialized) {
			classifier = ArticleClassifierService.getArticleClassifier(false);

			conn = getConnectionFactory().newConnection();
			channel = conn.createChannel();

			HashMap<String, Object> queueArgs = new HashMap<String, Object>();
			queueArgs.put("x-dead-letter-exchange", FAILURES_EXCHANGE_NAME);
			queueArgs.put("x-dead-letter-routing-key", FAILURES_ROUTING_KEY);

			AMQP.Queue.DeclareOk declareOk = channel.queueDeclare(INLET_QUEUE_NAME, true, false, false, queueArgs);

			AMQP.Queue.BindOk bindOk = channel.queueBind(declareOk.getQueue(), INLET_EXCHANGE_NAME, INLET_ROUTING_KEY);
			objectInitialized = true;
		}
	}

	public void launchQueue() throws IOException, InterruptedException {
		QueueingConsumer consumer = new QueueingConsumer(channel);
		channel.basicConsume(INLET_QUEUE_NAME, false, consumer);
		while (true) {
			QueueingConsumer.Delivery delivery = consumer.nextDelivery();

			if (processDelivery(delivery)) {
				channel.basicNack(delivery.getEnvelope().getDeliveryTag(), false, false);
			}
		}
	}

	private Boolean processDelivery(QueueingConsumer.Delivery delivery) throws IOException {
		JSONObject obj = new JSONObject(delivery.getBody());
		if (obj.isNull("title") || obj.isNull("wikitext")){
			return false;
		}

		Article art = new Article();
		art.setTitle(obj.getString("title"));
		art.setWikiText(obj.getString("wikitext"));
		try {
			String result = classifier.calssifySingleChoise(art);
			//FIXME: too generic exception
		} catch (Exception ex) {
			System.err.println(ex);
			return false;
		}
	}

	public static void main(String[] args) throws Exception {
		RabbitHole rabbit = new RabbitHole();
		rabbit.init();

		rabbit.launchQueue();

		rabbit.close();
	}
}