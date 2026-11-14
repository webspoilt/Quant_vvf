// ==============================================================================
// Hardware Acceleration: FPGA Top-of-Book BBO and Pre-Trade Risk Gate Module
// Designed for Xilinx Vivado HLS / Verilog Synthesis (Sub-50ns Pipeline)
// ==============================================================================
`timescale 1ns / 1ps

module hft_fpga_bbo_risk_engine #(
    parameter DATA_WIDTH = 64,
    parameter PRICE_WIDTH = 32,
    parameter QTY_WIDTH = 32
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // AXI-Stream Input from Network MAC/PHY Ingestion (Kernel Bypass)
    input  wire [DATA_WIDTH-1:0]  s_axis_tdata,
    input  wire                   s_axis_tvalid,
    output wire                   s_axis_tready,

    // Order Book Feed Parsed Signals
    input  wire                   in_tick_valid,
    input  wire                   in_tick_is_bid,
    input  wire [PRICE_WIDTH-1:0] in_tick_price,
    input  wire [QTY_WIDTH-1:0]   in_tick_qty,

    // Risk Bounds Configuration Registers
    input  wire [QTY_WIDTH-1:0]   cfg_max_clip_qty,
    input  wire [QTY_WIDTH-1:0]   cfg_max_gross_exposure,
    input  wire [PRICE_WIDTH-1:0] cfg_max_collar_spread,

    // Outbound Order Candidates to Evaluate
    input  wire                   eval_order_valid,
    input  wire                   eval_order_side,     // 0 = Buy, 1 = Sell
    input  wire [PRICE_WIDTH-1:0] eval_order_price,
    input  wire [QTY_WIDTH-1:0]   eval_order_qty,

    // Deterministic HW Output (< 2 Clock Cycles = ~6.4ns at 312.5 MHz)
    output reg                    order_approved,
    output reg                    order_rejected,
    output reg [2:0]              reject_reason,       // 1: Clip, 2: Gross, 3: Collar
    output reg [PRICE_WIDTH-1:0]  hw_best_bid,
    output reg [PRICE_WIDTH-1:0]  hw_best_ask
);

    assign s_axis_tready = 1'b1; // Never backpressure wire frames

    reg [QTY_WIDTH-1:0] current_gross_exposure;

    // 1. Hardware BBO Maintenance Pipeline (1 cycle)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hw_best_bid <= 0;
            hw_best_ask <= {PRICE_WIDTH{1'b1}};
        end else if (in_tick_valid) begin
            if (in_tick_is_bid) begin
                if (in_tick_price > hw_best_bid || in_tick_qty == 0)
                    hw_best_bid <= in_tick_price;
            end else begin
                if (in_tick_price < hw_best_ask || in_tick_qty == 0)
                    hw_best_ask <= in_tick_price;
            end
        end
    end

    // 2. Hardware Inline Pre-Trade Risk Gate (Single Clock Cycle Execution)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            order_approved         <= 1'b0;
            order_rejected         <= 1'b0;
            reject_reason          <= 3'b000;
            current_gross_exposure <= 0;
        end else if (eval_order_valid) begin
            // Check 1: Max Order Clip
            if (eval_order_qty > cfg_max_clip_qty || eval_order_qty == 0) begin
                order_approved <= 1'b0;
                order_rejected <= 1'b1;
                reject_reason  <= 3'b001; // CLIP VIOLATION
            // Check 2: Gross Exposure Limit
            end else if ((current_gross_exposure + eval_order_qty) > cfg_max_gross_exposure) begin
                order_approved <= 1'b0;
                order_rejected <= 1'b1;
                reject_reason  <= 3'b010; // GROSS EXPOSURE BREACH
            // Check 3: Hardware Price Collar
            end else if (eval_order_side == 1'b0 && (eval_order_price > (hw_best_ask + cfg_max_collar_spread))) begin
                order_approved <= 1'b0;
                order_rejected <= 1'b1;
                reject_reason  <= 3'b011; // BUY COLLAR VIOLATION
            end else if (eval_order_side == 1'b1 && (eval_order_price < (hw_best_bid - cfg_max_collar_spread))) begin
                order_approved <= 1'b0;
                order_rejected <= 1'b1;
                reject_reason  <= 3'b100; // SELL COLLAR VIOLATION
            // All checks passed: Approve instantaneously
            end else begin
                order_approved         <= 1'b1;
                order_rejected         <= 1'b0;
                reject_reason          <= 3'b000;
                current_gross_exposure <= current_gross_exposure + eval_order_qty;
            end
        end else begin
            order_approved <= 1'b0;
            order_rejected <= 1'b0;
        end
    end

endmodule
