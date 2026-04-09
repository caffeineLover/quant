from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence
import pandas as pd





@dataclass
class CashFlow:
    valuation_date: date
    currency: str
    name: str = ""
    data: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["date", "amount", "code"])
    )

    INT = "INT"
    PRN = "PRN"
    FEE = "FEE"
    DIV = "DIV"
    OTHER = "OTHER"

    def add_flow(self, flow_date: date, amount: float, code: str) -> None:
        self.data.loc[len(self.data)] = [flow_date, amount, code]

    def sort(self) -> None:
        self.data = self.data.sort_values("date").reset_index(drop=True)

    def add_discount_factors(self, discount_factors: Sequence[float]) -> None:
        self.data["discount_factor"] = list(discount_factors)

    def add_present_values(self) -> None:
        self.data["present_value"] = (
            self.data["amount"] * self.data["discount_factor"]
        )

    def npv(self) -> float:
        if "present_value" in self.data.columns:
            return float(self.data["present_value"].sum())

        if "discount_factor" in self.data.columns:
            return float((self.data["amount"] * self.data["discount_factor"]).sum())

        raise ValueError("No discount_factor or present_value column found.")

    def filter_code(self, code: str) -> pd.DataFrame:
        return self.data[self.data["code"] == code].copy()

    def interest_flows(self) -> CashFlow:
        df = self.filter_code(self.INT).reset_index(drop=True)
        return CashFlow(
            valuation_date=self.valuation_date,
            currency=self.currency,
            name=f"{self.name}:INT" if self.name else "INT",
            data=df,
        )

    def principal_flows(self) -> CashFlow:
        df = self.filter_code(self.PRN).reset_index(drop=True)
        return CashFlow(
            valuation_date=self.valuation_date,
            currency=self.currency,
            name=f"{self.name}:PRN" if self.name else "PRN",
            data=df,
        )