import mongoose from "mongoose";

const subscriptionSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User", 
      required: true,
    },
    plan: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Plan",
      required: true,
    },
    startDate: {
      type: Date,
      default: Date.now,
    },
    endDate: {
      type: Date,
      required: true,
    },
    status: {
      type: String,
      enum: ["active", "expired", "cancelled"],
      default: "active",
    },
    paymentId: {
      type: String, 
    },
  },
  { timestamps: true }
);

subscriptionSchema.methods.checkStatus = function () {
  if (this.endDate < new Date() && this.status === "active") {
    this.status = "expired";
  }
  return this.status;
};

const Subscription = mongoose.model("Subscription", subscriptionSchema);


export { Subscription };
