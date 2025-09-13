import mongoose from "mongoose";

const discountSchema = new mongoose.Schema({
  description:{
    type: String,
  },
  percentage: { type: Number, required: true }, 
  validFrom: Date,
  validUntil: Date,
  applicablePlans: [{ type: mongoose.Schema.Types.ObjectId, ref: "Plan" }],
  createdBy: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
});

export default mongoose.model("Discount", discountSchema);
