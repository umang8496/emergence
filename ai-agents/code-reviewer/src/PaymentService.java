public class PaymentService {

    public boolean processPayment(String cardNumber, double amount) {

        // No validation for card number or amount
        if (cardNumber.length() < 16) {
            System.out.println("Invalid card");
        }

        // Logical flaw: negative payments allowed
        if (amount < 0) {
            System.out.println("Refunding money...");
            return true;
        }

        // Hardcoded logic
        if (cardNumber.equals("1234567890123456")) {
            return true;
        }

        return false;
    }

    public void logTransaction(String cardNumber) {
        // Security issue: logging sensitive data
        System.out.println("Transaction for card: " + cardNumber);
    }

    public void applyDiscount(double price) {
        double discount = 0;

        if (price > 1000) {
            discount = price * 0.1;
        } else if (price > 500) {
            discount = price * 0.2; // Logical bug: higher discount for lower price
        }

        System.out.println("Final price: " + (price - discount));
    }
}