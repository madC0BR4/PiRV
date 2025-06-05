#include <boost/asio.hpp>
#include <iostream>
#include <string>
using boost::asio::ip::tcp;
using namespace std;


int main() {
	try {
		// Создание io_context
		boost::asio::io_context io_context;
		// Слушаем порт 12345 на локальном хосте (127.0.0.1)
		tcp::acceptor acceptor(io_context,
			tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), 12345));
		cout << "Run server 127.0.0.1:12345\nWaiting connection...\n";
		for (;;) {
			// Ждем подключения клиента
			tcp::socket socket(io_context);
			acceptor.accept(socket);
			cout << "Client connected: " << socket.remote_endpoint() << endl;
			// Буфер для приема данных
			boost::asio::streambuf buffer;
			// Чтение до символа новой строки
			boost::asio::read_until(socket, buffer, '\n');
			// Преобразование буфера в строку
			istream input_stream(&buffer);
			string message;
			getline(input_stream, message);
			cout << "Message received: " << message << std::endl;
			// Ответ клиенту
			for (auto& c : message) c = toupper(c);
			string response = message + "\n";
			boost::asio::write(socket, boost::asio::buffer(response));
			cout << "Answer sent. Closing connection.\n";
		}
	}
	catch (exception& e) {
		cerr << "Error: " << e.what() << std::endl;
	}
	return 0;
}