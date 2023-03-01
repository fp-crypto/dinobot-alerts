import ape
from fastapi import FastAPI, Response
import uvicorn
import telebot

import hmac
import hashlib
import string
from datetime import datetime
from os import environ

from ape import chain, project, networks


# start ape network connect
geth_url = environ["GETH_URL"]
networks.parse_network_choice(f"ethereum:mainnet:{geth_url}").__enter__()
signing_key = ""

app = FastAPI()

telegram_bot_key = environ["TELEGRAM_BOT_KEY"]
bot = telebot.TeleBot(telegram_bot_key)

etherscan_base_url = "https://etherscan.io/"

trade_handler = "0xb634316E06cC0B358437CbadD4dC94F1D3a92B3b"  #'0xcADBA199F3AC26F67f660C89d43eB1820b7f7a3b'
barn_solver = "0x8a4e90e9AFC809a69D2a3BDBE5fff17A12979609"
prod_solver = "0x398890BE7c4FAC5d766E1AEFFde44B2EE99F38EF"
settlement = project.Settlement.at("0x9008d19f58aabd9ed0d60971565aa8510560ab41")
oracle = project.ORACLE.at("0x83d95e0D5f402511dB06817Aff3f9eA88224B030")
alerts_enabled = True

CHAT_IDS = {
    "WAVEY_ALERTS": "-789090497",
    "GNOSIS_CHAIN_POC": "-1001516144118",
}


@app.get("/")
async def root():
    return {"message": "Hello World"}


from pydantic import BaseModel


class Alert(BaseModel):
    id: str
    event_type: str | None = None
    transaction: dict


@app.post("/solver/solve", status_code=200, response_class=Response)
async def alert_solver_solve(alert: Alert):

    txn = alert.transaction
    receipt = networks.provider.get_receipt(txn["hash"])
    logs = receipt.decode_logs([settlement.Settlement])

    for l in logs:
        txn_hash = l.transaction_hash
        solver = l.dict()["event_arguments"]["solver"]
        if solver not in [barn_solver, prod_solver]:
            continue
        block = l.block_number
        trades = enumerate_trades(receipt)
        slippage = calculate_slippage(trades, block)
        format_solver_alert(solver, txn_hash, block, trades, slippage)


def calculate_slippage(trades, block):

    slippages = {}
    for trade in trades:
        buy_token_address = trade["buy_token_address"]

        # If there is a trade for eth, use weth instead since TH will never
        # get native eth
        if (
            buy_token_address.lower()
            == "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE".lower()
        ):
            buy_token_address = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

        # we might have calculated the slippage previously
        if buy_token_address in slippages:
            continue

        buy_token = project.ERC20.at(buy_token_address)
        before = buy_token.balanceOf(trade_handler, block_identifier=block - 1)
        after = buy_token.balanceOf(trade_handler, block_identifier=block + 1)
        slippages[buy_token_address] = after - before

    return slippages


def enumerate_trades(receipt):
    logs = receipt.decode_logs([settlement.Trade])

    trades = []
    for l in logs:
        args = l.dict()["event_arguments"]
        eth = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
        try:
            sell_token = args["sellToken"]
            if sell_token == eth:
                sell_token_symbol = "ETH"
                sell_token_decimals = 18
            else:
                sell_token_symbol = project.ERC20.at(sell_token).symbol()
                sell_token_decimals = project.ERC20.at(sell_token).decimals()
        except:
            sell_token_symbol = "? Cannot Find ?"
            if sell_token == "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2":
                sell_token_symbol = "MKR"
            sell_token_decimals = 18

        try:
            buy_token = args["buyToken"]
            if buy_token == eth:
                buy_token_symbol = "ETH"
                buy_token_decimals = 18
            else:
                buy_token_symbol = project.ERC20.at(buy_token).symbol()
                buy_token_decimals = project.ERC20.at(buy_token).decimals()
        except:
            buy_token_symbol = "? Cannot Find ?"
            if buy_token == "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2":
                buy_token_symbol = "MKR"
            buy_token_decimals = 18

        trade = {
            "owner": args["owner"],
            "sell_token_address": args["sellToken"],
            "sell_token_symbol": sell_token_symbol,
            "sell_token_decimals": sell_token_decimals,
            "buy_token_address": args["buyToken"],
            "buy_token_symbol": buy_token_symbol,
            "buy_token_decimals": buy_token_decimals,
            "sell_amount": args["sellAmount"],
            "buy_amount": args["buyAmount"],
            "fee_amount": args["feeAmount"],
            "order_uid": "0x" + args["orderUid"].hex(),
        }
        trades.append(trade)
    return trades


def format_solver_alert(solver, txn_hash, block, trade_data, slippages):

    prod_solver = "0x398890BE7c4FAC5d766E1AEFFde44B2EE99F38EF"
    cow_explorer_url = f'https://explorer.cow.fi/orders/{trade_data[0]["order_uid"]}'
    cow_explorer_url = f"https://explorer.cow.fi/tx/{txn_hash}"
    ethtx_explorer_url = f"https://ethtx.info/mainnet/{txn_hash}"
    tonkers_base_url = f"https://prod.seasolver.dev/1/route/"
    eigen_url = f"https://eigenphi.io/mev/eigentx/{txn_hash}"
    barn_solver = "0x8a4e90e9AFC809a69D2a3BDBE5fff17A12979609"
    if solver == barn_solver:
        tonkers_base_url = f"https://barn.seasolver.dev/1/route/"
    txn_receipt = networks.provider.get_receipt(txn_hash)
    ts = chain.blocks[block].timestamp
    index = get_index_in_block(txn_hash)
    index = index if index != 1_000_000 else "???"

    dt = datetime.utcfromtimestamp(ts).strftime("%m/%d %H:%M")
    msg = f'{"🧜‍♂️" if solver == prod_solver else "🐓"} *New solve detected!*\n'
    msg += f"by [{solver[0:7]}...]({etherscan_base_url}address/{solver})  index: {index} @ {dt}\n\n"
    msg += f"📕 *Trade(s)*:\n"
    for t in trade_data:
        user = t["owner"]
        sell_amt = round(t["sell_amount"] / 10 ** t["sell_token_decimals"], 4)
        buy_amt = round(t["buy_amount"] / 10 ** t["buy_token_decimals"], 4)
        buy_token = t["buy_token_address"]
        if buy_token.lower() == "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE".lower():
            buy_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        msg += f'    [ 🐻‍❄️ ]({tonkers_base_url}{t["sell_token_address"]}/{buy_token}) | [{t["sell_token_symbol"]}]({etherscan_base_url}token/{t["sell_token_address"]}) {sell_amt:,} -> [{t["buy_token_symbol"]}]({etherscan_base_url}token/{t["buy_token_address"]}) {buy_amt:,} | [{user[0:7]}...]({etherscan_base_url}address/{user})\n'
    msg += "\n✂️ *Slippages*"
    for key in slippages:
        token = project.ERC20.at(key)
        slippage = slippages[key]
        color = "🔴" if slippage < 0 else "🟢"
        amount = round(slippage / 10 ** token.decimals(), 4)
        try:
            msg += f"\n   {color} {token.symbol()}: {amount}"
        except:
            msg += f"\n   {color} -SymbolError-: {amount}"
    msg += f"\n\n{calc_gas_cost(txn_receipt)}"
    msg += f"\n\n🔗 [Etherscan]({etherscan_base_url}tx/{txn_hash}) | [Cow]({cow_explorer_url}) | [Eigen]({eigen_url}) | [EthTx]({ethtx_explorer_url})"

    # Add slippage info

    if alerts_enabled:
        chat_id = CHAT_IDS["GNOSIS_CHAIN_POC"]
    else:
        chat_id = CHAT_IDS["WAVEY_ALERTS"]
    bot.send_message(chat_id, msg, parse_mode="markdown", disable_web_page_preview=True)


def calc_gas_cost(txn_receipt):
    eth_used = txn_receipt.gas_price * txn_receipt.gas_used
    gas_cost = (
        oracle.getNormalizedValueUsdc(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", eth_used
        )
        / 10**6
    )
    return f"💸 ${round(gas_cost,2):,} | {round(eth_used/1e18,4)} ETH"


def get_index_in_block(txn_hash):
    tx = chain.provider.get_receipt(txn_hash)
    hashes = [x.txn_hash.hex() for x in chain.blocks[tx.block_number].transactions]
    try:
        return hashes.index(tx.txn_hash)
    except:
        return 1_000_000  # Not found


def isValidSignature(signature: string, body: bytes, timestamp: string):
    h = hmac.new(str.encode(signing_key), body, hashlib.sha256)
    h.update(str.encode(timestamp))
    digest = h.hexdigest()
    return hmac.compare_digest(signature, digest)


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0")
