import mongoose from "mongoose";

const planSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true,
    },
    description: {
      type: String,
      required: true,
    },
    price: {
      type: Number,
      required: true,
    },
    billingCycle: {
      type: String,
      enum: ["monthly", "quarterly", "semi-annual", "annual"],
      default: "monthly",
    },
    features: [{
      type: String,
      required: true,
    }],
    trialPeriod: {
      type: Number,  // Number of days
      default: 0,
    },
    tier: {
      type: String,
      enum: ["basic", "standard", "premium", "enterprise"],
      required: true,
    },
    status: {
      type: String,
      enum: ["active", "inactive", "deprecated"],
      default: "active",
    },
    allowDowngrade: {
      type: Boolean,
      default: true,
    },
    allowUpgrade: {
      type: Boolean,
      default: true,
    },
    downgradeTo: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: "Plan",
    }],
    upgradeTo: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: "Plan",
    }],
    earlyTerminationFee: {
      type: Number,
      default: 0,
    },
    autoRenewal: {
      type: Boolean,
      default: true,
    },
    cancellationPolicy: {
      noticePeriod: {
        type: Number,  
        default: 0,
      },
      refundPolicy: {
        type: String,
        enum: ["full", "prorated", "none"],
        default: "prorated",
      },
    },
  },
  { timestamps: true }
);


const Plan = mongoose.model("Plan", planSchema);

export default Plan;
