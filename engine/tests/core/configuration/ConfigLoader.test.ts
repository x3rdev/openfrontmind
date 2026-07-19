import { describe, expect, test } from "vitest";
import { GameEnv, parseGameEnv } from "../../../core/configuration/Config";

describe("parseGameEnv", () => {
  test("maps 'dev' to GameEnv.Dev", () => {
    expect(parseGameEnv("dev")).toBe(GameEnv.Dev);
  });
  test("maps 'staging' to GameEnv.Preprod", () => {
    expect(parseGameEnv("staging")).toBe(GameEnv.Preprod);
  });
  test("maps 'prod' to GameEnv.Prod", () => {
    expect(parseGameEnv("prod")).toBe(GameEnv.Prod);
  });
  test("throws on undefined", () => {
    expect(() => parseGameEnv(undefined)).toThrow(/unsupported game env/);
  });
  test("throws on unknown value", () => {
    expect(() => parseGameEnv("production")).toThrow(/unsupported game env/);
  });
});
