"""Generate the synthetic labelled dataset data/support_tickets.csv (90 tickets, 30 per class).

The data is synthetic so the project can be shared publicly. In a real deployment this file is
replaced by an export of historical tickets with their human-confirmed category.

Usage:  python scripts/generate_sample_tickets.py
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "support_tickets.csv"

BILLING = [
    ("Invoice charged twice", "Hi, I was charged twice for invoice 4471 this month. Please refund the duplicate payment."),
    ("Wrong VAT on invoice", "The VAT on our last invoice is 19% but we are VAT exempt. Please send a corrected invoice."),
    ("Refund request", "We returned the filter kit three weeks ago and still have not received the refund."),
    ("Payment not reflected", "We paid by bank transfer on Monday but your portal still shows the invoice as unpaid."),
    ("Credit note missing", "Please issue the credit note for the cancelled service visit."),
    ("Update billing address", "Our finance department moved. Please update the billing address on all future invoices."),
    ("Price higher than quote", "The invoice amount is higher than the quotation we approved. Can you explain the difference?"),
    ("Duplicate invoice", "We received the same invoice number twice with different amounts."),
    ("Payment terms", "Can we change our payment terms from 30 days to 60 days?"),
    ("Late payment fee", "Why was a late payment fee added? We paid before the due date."),
    ("Invoice copy", "Please resend a PDF copy of invoice 9932 for our auditors."),
    ("Service contract billing", "Our service contract was billed monthly but the agreement says quarterly billing."),
    ("Card declined", "My credit card payment for the spare parts order was declined, how else can I pay?"),
    ("Currency on invoice", "The invoice is in USD but our contract is in EUR."),
    ("Overcharged for parts", "We were overcharged for the oil separator, the price list says less."),
    ("Refund status", "What is the status of the refund for order 55120?"),
    ("Invoice not received", "We have not received the invoice for the September maintenance visit."),
    ("Wrong purchase order on invoice", "The purchase order number on the invoice is wrong so our system rejects it."),
    ("Split payment", "Can we split the payment for the new compressor into two instalments?"),
    ("Bank details for payment", "Please confirm your bank details so we can pay the outstanding invoice."),
    ("Discount not applied", "Our agreed 10% discount was not applied on the last bill."),
    ("Tax invoice format", "We need the tax invoice in the new e-invoice format for GST."),
    ("Billing contact change", "Please send all invoices to our new accounts payable team."),
    ("Extra charge for travel", "Why is there an extra travel charge on the field service invoice?"),
    ("Prepayment refund", "We cancelled the order, please refund the prepayment."),
    ("Statement of account", "Can you send our statement of account for this year?"),
    ("Payment reminder error", "We got a payment reminder for an invoice that was already settled."),
    ("Invoice amount dispute", "We dispute the invoice amount for the rental compressor."),
    ("Wrong customer name on invoice", "Our company name is spelled wrong on the invoice."),
    ("Recurring charge", "There is a recurring monthly charge we do not recognise on our account."),
]

TECHNICAL = [
    ("Compressor stops with error", "Our AX-200 compressor stops after ten minutes and shows error E104."),
    ("High discharge temperature", "The controller shows a high discharge temperature alarm every afternoon."),
    ("Machine not starting", "After the maintenance visit the machine does not start at all."),
    ("Pressure drop", "System pressure drops below the set point when two tools run together."),
    ("Oil leak", "There is an oil leak under the compressor near the drain valve."),
    ("Strange noise", "The motor makes a loud rattling noise during start-up."),
    ("Controller display blank", "The controller display is blank but the unit seems to have power."),
    ("Firmware update failed", "The firmware update on the controller failed halfway and now it reboots."),
    ("Dryer alarm", "The refrigerant dryer shows a dew point alarm."),
    ("Installation question", "What clearance is needed around the compressor during installation?"),
    ("Sensor fault", "We get a pressure sensor fault code on the display."),
    ("Unit trips", "The unit trips the main breaker at start-up."),
    ("Overheating", "The compressor overheats in summer even though the room is ventilated."),
    ("Remote monitoring offline", "The remote monitoring gateway shows the machine offline since yesterday."),
    ("Water in air line", "There is water in the compressed air line after the dryer."),
    ("Vibration", "Excessive vibration on the vacuum pump after replacing the coupling."),
    ("Service interval", "The service interval warning appears even though service was done last week."),
    ("Motor overload", "Motor overload alarm appears when the machine runs at full load."),
    ("Pressure setting", "How do I change the unloading pressure setting on the controller?"),
    ("Safety valve blowing", "The safety valve opens and blows off air during normal operation."),
    ("Spare part for repair", "Which spare part number do we need to replace the broken inlet valve?"),
    ("Power fluctuation shutdown", "The machine shuts down every time there is a power fluctuation."),
    ("High energy consumption", "Energy consumption doubled this month without any change in production."),
    ("Error code meaning", "What does alarm code E212 mean on the AX controller?"),
    ("Air filter warning", "The air filter warning stays on after replacing the filter."),
    ("Cannot connect to controller", "Our engineer cannot connect the laptop to the controller service port."),
    ("Vacuum level low", "The vacuum pump does not reach the required vacuum level."),
    ("Fan not running", "The cooling fan does not run although the compressor is loaded."),
    ("Leak detection", "How can we find air leaks in our compressed air network?"),
    ("Drain valve stuck", "The automatic drain valve seems stuck open and air is escaping."),
]

GENERAL = [
    ("Opening hours", "What are your office opening hours during the holidays?"),
    ("Product catalogue", "Where can I download the latest product catalogue?"),
    ("Distributor in Pune", "Do you have an authorised distributor in Pune?"),
    ("Factory visit", "We would like to visit your factory with our engineering team."),
    ("Account manager", "Who is our account manager for the western region?"),
    ("Training courses", "Do you offer training courses for maintenance staff?"),
    ("Partnership enquiry", "We are interested in becoming a sales partner in our region."),
    ("Career opportunities", "Are there any open positions in your service department?"),
    ("Sustainability report", "Where can I find your latest sustainability report?"),
    ("Trade fair", "Will you exhibit at the industrial trade fair next month?"),
    ("Contact for press", "Who should journalists contact for press enquiries?"),
    ("Brochure request", "Please send us a brochure about your vacuum solutions."),
    ("Office address", "What is the address of your regional office?"),
    ("Newsletter", "How can I subscribe to your product newsletter?"),
    ("Product comparison", "What is the difference between your fixed speed and variable speed ranges?"),
    ("Warranty information", "How long is the standard warranty on new compressors?"),
    ("Demo request", "Can someone show us a demo of the new tool range?"),
    ("Certificates", "Can you share your ISO certificates for our supplier audit?"),
    ("Supplier registration", "How do we register as a supplier to your company?"),
    ("Event invitation", "Thank you for the invitation, can two colleagues join the customer event?"),
    ("Website login", "How do I create an account on your customer portal?"),
    ("Lead time question", "What is the usual delivery lead time for new machines?"),
    ("Recycling old machine", "Do you take back old compressors for recycling?"),
    ("Language support", "Is your support team available in Hindi and Marathi?"),
    ("Feedback", "We want to give positive feedback about your service engineer."),
    ("Rental options", "Do you offer machines for short term rental?"),
    ("Case study", "Do you have case studies from the food industry?"),
    ("Holiday closure", "Is your support line open on public holidays?"),
    ("Information request", "We need general information about your company for a vendor form."),
    ("Meeting request", "Can we schedule a meeting to discuss future projects?"),
]


def main() -> None:
    """Write 90 tickets (30 per class) to data/support_tickets.csv with interleaved dates."""
    start = datetime.date(2026, 1, 5)
    rows = []
    # Interleave classes over time so both the older (train) and newer (test) split contain every class.
    for i in range(30):
        for offset, (examples, label) in enumerate([(BILLING, "billing"), (TECHNICAL, "technical"), (GENERAL, "general")]):
            subject, body = examples[i]
            received = start + datetime.timedelta(days=i * 5 + offset)
            rows.append({"received_at": received.isoformat(), "subject": subject, "body": body, "category": label})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["received_at", "subject", "body", "category"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} tickets to {OUT}")


if __name__ == "__main__":
    main()
