#include <boost/asio.hpp>
#include <iostream>
#include <string>
using boost::asio::ip::tcp;
using namespace std;
int main() {
	try {
		// Контекст ввода/вывода
		boost::asio::io_context io_context;
		// Создаем сокет
		tcp::socket socket(io_context);
		// Устанавливаем соединение с сервером
		tcp::endpoint endpoint(boost::asio::ip::make_address("127.0.0.1"), 12345);
		socket.connect(endpoint);
		cout << "Connection to the server established.\n";
		// Сообщение для отправки
		string message;
		cout << "Enter message: ";
		getline(cin,message);
		message += "\n";
		boost::asio::write(socket, boost::asio::buffer(message));
		cout << "Message sent: " << message;
		// Буфер для приема ответа
		boost::asio::streambuf buffer;
		boost::asio::read_until(socket, buffer, '\n');
		// Чтение и вывод ответа
		istream input_stream(&buffer);
		string response;
		getline(input_stream, response);
		cout << "Answer from server: " << response << endl;
	}
	catch (std::exception& e) {
		cerr << "Client error: " << e.what() << endl;
	}
	return 0;
}