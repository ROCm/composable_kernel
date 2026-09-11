package org.ck

class TransientFault extends Exception {
    TransientFault(String reason) { super(reason) }
}
